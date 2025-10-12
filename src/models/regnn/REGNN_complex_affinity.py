import torch
from pytorch_lightning import LightningModule
from torch.nn import Linear, ReLU
from torch.nn import functional as F
from torch.optim import Adam
from torch_geometric.nn.pool import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn.norm import BatchNorm, LayerNorm
from scipy.stats import pearsonr, spearmanr
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR

from .regnn_ensemble import REGCLEnsemble

class REGNN_complex_affinity(LightningModule):
    """
    Equivariant Relational Graph Neural Network for predicting protein-ligand binding affinity.
    Uses REGCLEnsemble to maintain equivariance while handling multiple relation types.
    """

    def __init__(self, in_dim, out_dim=1, hidden_dim=64, lr=1e-4, num_relations=2,
                 edge_attr_dim=1, num_gnn_layers=5, num_fc_layers=2, dropout=0.1,
                 pooling='mean', node_agg='sum', coords_agg='mean', attention=False,
                 normalize=False, tanh=False, use_edge_attr=True, norm='bn',
                 residual=True, use_ls=False, ex_ls_gamma=1.0, log_every_step=False, update_coords=False):
        super(REGNN_complex_affinity, self).__init__()
        # Add this embedding layer
        self.embedding = Linear(in_dim, hidden_dim)        
        self.save_hyperparameters()
        
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        self.num_relations = num_relations
        self.lr = lr
        self.norm = norm
        self.use_edge_attr = use_edge_attr
        self.dropout = dropout
        self.pooling = pooling
        self.use_ls = use_ls
        self.ex_ls_gamma = ex_ls_gamma
        self.log_every_step = log_every_step
        self.update_coords = update_coords
        
        # GNN layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        # Input layer
        self.convs.append(REGCLEnsemble(
            input_nf=hidden_dim,
            hidden_nf=hidden_dim,
            output_nf=hidden_dim,
            edges_in_d=edge_attr_dim if use_edge_attr else 0,
            num_relations=num_relations,
            act_fn=torch.nn.SiLU(),
            node_agg=node_agg,
            coords_agg=coords_agg,
            attention=attention,
            normalize=normalize,
            tanh=tanh,
            residual=residual,
            update_coords=update_coords
        ))
        
        if norm == 'bn':
            self.bns.append(BatchNorm(hidden_dim))
        elif norm == 'ln':
            self.bns.append(LayerNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(num_gnn_layers - 1):
            self.convs.append(REGCLEnsemble(
                input_nf=hidden_dim,
                hidden_nf=hidden_dim,
                output_nf=hidden_dim,
                edges_in_d=edge_attr_dim if use_edge_attr else 0,
                num_relations=num_relations,
                act_fn=torch.nn.SiLU(),
                node_agg=node_agg,
                coords_agg=coords_agg,
                attention=attention,
                normalize=normalize,
                tanh=tanh,
                residual=residual,
                update_coords=update_coords
            ))
            if norm == 'bn':
                self.bns.append(BatchNorm(hidden_dim))
            elif norm == 'ln':
                self.bns.append(LayerNorm(hidden_dim))
        
        # Rest of the code remains unchanged...
        # Fully connected layers
        self.fcs = torch.nn.ModuleList()
        fc_input_dim = hidden_dim * 2  # Concatenated protein and ligand representations

        # All hidden layers use the same width (fc_input_dim)
        for _ in range(num_fc_layers - 1):
            self.fcs.append(Linear(fc_input_dim, fc_input_dim))

        # Only the output layer reduces to final dimension
        self.fcs.append(Linear(fc_input_dim, out_dim))
        
        # Non-linear activation
        self.relu = ReLU()
        
        # Dropout layer
        self.dropout_layer = torch.nn.Dropout(p=dropout)
        
        # Pooling function
        if self.pooling == "mean":
            self.pooling_layer = global_mean_pool
        elif self.pooling == "add":
            self.pooling_layer = global_add_pool
        elif self.pooling == "max":
            self.pooling_layer = global_max_pool

    def forward(self, batch):
        x, pos = batch.x, batch.pos
        edge_index = batch.edge_index
        edge_type = batch.edge_type
        edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') and self.use_edge_attr else None
        
        # Get protein/ligand indicator (last feature dimension)
        indices = x[:, -1].clone().long()
        # Apply embedding to input features
        x = self.embedding(x)
    
        # Process through GNN layers (use full feature vector)
        for conv, bn in zip(self.convs, self.bns):
            h, pos = conv(x, edge_index, pos, edge_attr, edge_type)
            h = F.relu(h)
            if self.training:
                h = self.dropout_layer(h)
            if self.norm == 'bn':
                h = bn(h)
            elif self.norm == 'ln':
                h = bn(h, batch=batch.batch)
            x = h  # Update x for next layer
        
        # Pooling
        x_prot = self.pooling_layer(x[indices == 0], batch=batch.batch[indices == 0])
        x_lig = self.pooling_layer(x[indices == 1], batch=batch.batch[indices == 1])
        
        # Concatenate protein and ligand representations
        out = torch.cat([x_prot, x_lig], dim=1)
        
        # FC layers
        for i, layer in enumerate(self.fcs[:-1]):
            out = layer(out)
            out = self.relu(out)
            if self.training:
                out = self.dropout_layer(out)
        
        # Output layer
        out = self.fcs[-1](out)
        
        return out

    def configure_optimizers(self):
        print("Learning rate:", self.lr)
        optimizer = Adam(self.parameters(), lr=self.lr)
        
        if self.use_ls:
            print(f"Using ExponentialLR. Gamma: {self.ex_ls_gamma}")
            scheduler = ExponentialLR(
                optimizer,
                gamma=self.ex_ls_gamma,
            )
            return [optimizer], [scheduler]
        
        return optimizer

    def training_step(self, batch, batch_idx):
        y_hat = self.forward(batch).view(-1)
        y = batch.y
        
        loss = F.mse_loss(y_hat, y)
        
        # Log with step control and consistent naming convention
        # self.log("train/y_hat_-logKdKi", y_hat[0], on_step=self.log_every_step, batch_size=y.shape[0])
        # self.log("train/y_-logKdKi", y[0], on_step=self.log_every_step, batch_size=y.shape[0])
        self.log("train/loss", loss, batch_size=y.shape[0], on_step=self.log_every_step, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.forward(batch).view(-1)
        y = batch.y
        
        loss = F.mse_loss(y_hat, y)
        
        self.log("val/loss", loss, batch_size=y.shape[0], add_dataloader_idx=False)
        self.validation_step_outputs.append((y_hat.detach().cpu(), y.detach().cpu()))
        
        return loss

    def test_step(self, batch, batch_idx):
        y_hat = self.forward(batch).view(-1)
        y = batch.y
        
        loss = F.mse_loss(y_hat, y)
        self.log("test/loss", loss, batch_size=y.shape[0], on_epoch=True)
        self.test_step_outputs.append((y_hat.detach().cpu(), y.detach().cpu()))
        
        return loss

    def predict_step(self, batch, batch_idx):
        y_hat = self.forward(batch).view(-1)
        y = batch.y
        return (y, y_hat, batch.pdb_id)

    def log_metrics(self, y_hats, ys, prefix):
        y_hats = torch.cat(y_hats)
        ys = torch.cat(ys)
        
        # Convert tensors to numpy for correlation calculations
        y_hats_np = y_hats.numpy()
        ys_np = ys.numpy()
        
        # Calculate Pearson and Spearman correlations
        pearson_corr = pearsonr(y_hats_np, ys_np)[0]
        spearman_corr = spearmanr(y_hats_np, ys_np)[0]
        
        # Calculate RMSE and MAE
        rmse = torch.sqrt(F.mse_loss(y_hats, ys))
        mae = F.l1_loss(y_hats, ys)
        
        # Log the metrics
        self.log(f"{prefix}/pearson_corr", pearson_corr, on_epoch=True)
        self.log(f"{prefix}/spearman_corr", spearman_corr, on_epoch=True)
        self.log(f"{prefix}/rmse", rmse, on_epoch=True)
        self.log(f"{prefix}/mae", mae, on_epoch=True)

    def on_validation_epoch_end(self):
        if len(self.validation_step_outputs) > 0:
            y_hats, ys = zip(*self.validation_step_outputs)
            self.log_metrics(y_hats, ys, "val")
            self.validation_step_outputs.clear()
    
    def on_test_epoch_end(self):
        if len(self.test_step_outputs) > 0:
            y_hats, ys = zip(*self.test_step_outputs)
            self.log_metrics(y_hats, ys, "test")
            self.test_step_outputs.clear()

    def on_validation_epoch_end(self):
        if len(self.validation_step_outputs) > 0:
            y_hats, ys = zip(*self.validation_step_outputs)
            self.log_metrics(y_hats, ys, "val")
            self.validation_step_outputs.clear()
import torch
from pytorch_lightning import LightningModule
from torch.nn import Linear, ReLU
from torch.nn import functional as F
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from scipy.stats import pearsonr, spearmanr
from torch_geometric.nn.norm import BatchNorm, LayerNorm

from .regnn_ensemble import REGCLEnsemble

class REGNNNodeRegressor(LightningModule):
    """
    Equivariant Relational Graph Neural Network for node regression.
    Uses REGCLEnsemble to maintain equivariance while handling multiple relation types.
    """

    def __init__(self, in_dim, hidden_dim=64, out_dim=1, num_relations=2, lr=1e-4,
                 num_gnn_layers=5, num_fc_layers=2, dropout=0.1, gnn_dropout=0.0,
                 node_agg='sum', coords_agg='mean', attention=False, normalize=False,
                 tanh=False, use_edge_attr=True, edge_attr_dim=1, norm='bn',
                 residual=True, log_every_step=False, update_coords=False):
        super(REGNNNodeRegressor, self).__init__()
        self.save_hyperparameters()
        # Add this embedding layer
        self.embedding = Linear(in_dim, hidden_dim)       
        self.num_relations = num_relations
        self.lr = lr
        self.dropout = dropout
        self.norm = norm
        self.use_edge_attr = use_edge_attr
        self.log_every_step = log_every_step
        self.update_coords = update_coords
        
        # Store step outputs for metrics calculation
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
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
        
        # FC layers
        self.fcs = torch.nn.ModuleList()
        for i in range(num_fc_layers - 1):
            self.fcs.append(Linear(hidden_dim, hidden_dim))
        
        # Output layer
        self.fcs.append(Linear(hidden_dim, out_dim))
        
        # Dropout layer
        self.dropout_layer = torch.nn.Dropout(p=dropout)
        
        # Loss function
        self.loss_fn = torch.nn.MSELoss()
        
        # Metrics
        self.mae = MeanAbsoluteError()
        self.rmse = MeanSquaredError(squared=False)
        
        print("The model REGCLNodeRegressor is initialized!")

    def forward(self, batch):
        x, pos = batch.x, batch.pos
        edge_index = batch.edge_index
        edge_type = batch.edge_type
        edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') and self.use_edge_attr else None

        # Apply embedding to input features
        x = self.embedding(x)        
        # Process through GNN layers
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
        
        # FC layers
        for layer in self.fcs[:-1]:
            x = layer(x)
            x = F.relu(x)
            if self.training:
                x = self.dropout_layer(x)
        
        # Output layer
        out = self.fcs[-1](x)
        
        return out

    def training_step(self, batch, batch_idx):
        y_hat = self.forward(batch).squeeze()
        y = batch.y
        
        loss = self.loss_fn(y_hat, y)
        
        # Store predictions and targets for epoch-end metrics
        self.train_step_outputs.append((y_hat.detach().cpu(), y.detach().cpu()))
        
        # Log loss with step control
        self.log("train/loss", loss, on_step=self.log_every_step, on_epoch=True, prog_bar=True, batch_size=y.shape[0])
        
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.forward(batch).squeeze()
        y = batch.y
        
        loss = self.loss_fn(y_hat, y)
        
        self.log("val/loss", loss, batch_size=y.shape[0], add_dataloader_idx=False)
        self.validation_step_outputs.append((y_hat.cpu(), y.cpu()))
        
        return loss

    def test_step(self, batch, batch_idx):
        y_hat = self.forward(batch).squeeze()
        y = batch.y
        
        loss = self.loss_fn(y_hat, y)
        self.log("test/loss", loss, batch_size=y.shape[0], on_epoch=True)
        self.test_step_outputs.append((y_hat.detach().cpu(), y.detach().cpu()))
        
        return loss

    def predict_step(self, batch, batch_idx):
        y_hat = self.forward(batch).squeeze()
        y = batch.y
        return (y, y_hat, batch.pdb_id)

    def log_metrics(self, y_hats, ys, prefix):
        y_hats = torch.cat(y_hats) if isinstance(y_hats, (list, tuple)) else y_hats
        ys = torch.cat(ys) if isinstance(ys, (list, tuple)) else ys
        
        # Convert to numpy for correlation calculations
        y_hats_np = y_hats.numpy()
        ys_np = ys.numpy()
        
        # Calculate Pearson and Spearman correlations
        pearson_corr, _ = pearsonr(y_hats_np, ys_np)
        spearman_corr, _ = spearmanr(y_hats_np, ys_np)
        
        # Calculate RMSE and MAE
        rmse = torch.sqrt(F.mse_loss(y_hats, ys))
        mae = F.l1_loss(y_hats, ys)
        
        # Log metrics
        self.log(f"{prefix}/pearson_corr", pearson_corr, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/spearman_corr", spearman_corr, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/rmse", rmse, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/mae", mae, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self):
        if len(self.train_step_outputs) > 0:
            y_hats, ys = zip(*self.train_step_outputs)
            y_hats = torch.cat(y_hats)
            ys = torch.cat(ys)
            
            # Convert to numpy for correlation calculations
            y_hats_np = y_hats.numpy()
            ys_np = ys.numpy()
            
            # Calculate Pearson correlation
            pearson_corr, _ = pearsonr(y_hats_np, ys_np)
            
            # Calculate RMSE and MAE
            rmse = torch.sqrt(F.mse_loss(y_hats, ys))
            mae = F.l1_loss(y_hats, ys)
            
            # Log metrics
            self.log("train/pearson_corr", pearson_corr, on_epoch=True, prog_bar=True)
            self.log("train/rmse", rmse, on_epoch=True, prog_bar=True)
            self.log("train/mae", mae, on_epoch=True, prog_bar=True)
            
            # Clear the list
            self.train_step_outputs.clear()

    def on_validation_epoch_end(self):
        if hasattr(self, 'validation_step_outputs') and len(self.validation_step_outputs) > 0:
            y_hats, ys = zip(*self.validation_step_outputs)
            self.log_metrics(y_hats, ys, "val")
            self.validation_step_outputs.clear()
    
    def on_test_epoch_end(self):
        if hasattr(self, 'test_step_outputs') and len(self.test_step_outputs) > 0:
            y_hats, ys = zip(*self.test_step_outputs)
            self.log_metrics(y_hats, ys, "test")
            self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
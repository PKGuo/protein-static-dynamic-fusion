import torch
from pytorch_lightning import LightningModule
from torch.nn import Linear, ReLU
from torch.nn import functional as F
from torch.optim import Adam
from torch_geometric.nn import RGCNConv
from torch_geometric.nn.norm import BatchNorm, LayerNorm
from torch_geometric.nn.pool import global_mean_pool, global_add_pool, global_max_pool
from scipy.stats import pearsonr, spearmanr
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR

class New_RGCN_complex_affinity(LightningModule):
    """
    Relational Graph Attention Network model class based on RGATConv for predicting single complex -logKd/Ki.
    """

    def __init__(self, in_dim, out_dim=1, hidden_dim=64, lr=1e-6, num_relations=2, heads=1,
             edge_attr_dim=1, num_gnn_layers=5, num_fc_layers=2, concat=False,
             use_edge_attr=False, gnn_dropout = 0.0, dropout=0.1, pooling='mean', norm='bn', root_weight=False, aggr='mean',use_ls=False, ex_ls_gamma=1.0):
        

        """
        Initialization for RGAT model.

        in_dim: int - The dimensionality of input node features.
        out_dim: int - The dimensionality of output (default: 1).
        hidden_dim: int - The number of hidden units in the hidden layers.
        lr: float - learning rate for the optimizer.
        num_relations: int - The number of relation types.
        heads: int - Number of attention heads.
        edge_dim: int - Edge feature dimensionality (if using edge_attr).
        num_gnn_layers: int - Number of GNN layers.
        num_fc_layers: int - Number of fully connected layers.
        concat: bool - Whether to concatenate the heads.
        """
        super(New_RGCN_complex_affinity, self).__init__()

        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.num_relations = num_relations
        self.lr = lr
        # self.heads = heads
        self.norm = norm

        # self.concat = concat
        self.use_edge_attr = False #use_edge_attr=False because currently rgcnconv does not support edge_attr
        self.dropout = dropout
        # self.gnn_dropout = gnn_dropout

        self.pooling = pooling

        if self.use_edge_attr:
            edge_dim = edge_attr_dim  # Set edge_dim to edge_attr dimensionality
        else:
            edge_dim = None  # Don't use edge_attr, edge_dim is None

        # Calculate the effective hidden dimension considering heads and concat
        # effective_hidden_dim = hidden_dim * heads if concat else hidden_dim
        effective_hidden_dim = hidden_dim
        # RGAT layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.aggr = aggr

        self.use_ls = use_ls
        self.ex_ls_gamma = ex_ls_gamma

        # Input layer
        self.convs.append(RGCNConv(in_dim, hidden_dim, num_relations=num_relations, root_weight=root_weight, aggr=self.aggr))
        if self.norm == 'bn':
            self.bns.append(BatchNorm(effective_hidden_dim))
        elif self.norm == 'ln':
            self.bns.append(LayerNorm(effective_hidden_dim))
        
        print("norm: ", self.norm)

        # Hidden layers
        for _ in range(num_gnn_layers - 1):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations, root_weight=root_weight, aggr=self.aggr))

            if self.norm == 'bn':
                self.bns.append(BatchNorm(effective_hidden_dim))
            elif self.norm == 'ln':
                self.bns.append(LayerNorm(effective_hidden_dim))

        # Fully connected layers
        self.fcs = torch.nn.ModuleList()
        fc_input_dim = effective_hidden_dim * 2  # Concatenated protein and ligand representations

        for _ in range(num_fc_layers - 1):
            self.fcs.append(Linear(fc_input_dim, fc_input_dim))
            # self.fcs.append(ReLU())

        # Output layer
        self.fcs.append(Linear(fc_input_dim, out_dim))

        # Non-linear activation
        self.relu = ReLU()

        self.dropout_layer = torch.nn.Dropout(p=self.dropout)

        if self.pooling == "mean":
            self.pooling_layer = global_mean_pool
        elif self.pooling == "add":
            self.pooling_layer = global_add_pool
        elif self.pooling == "max":
            self.pooling_layer = global_max_pool

    def forward(self, batch):
        """
        Forward pass for the RGAT model for a single complex.

        batch: Batch - The batch containing all the necessary attributes.
        """
        x, edge_index, edge_attr, edge_type = batch.x, batch.edge_index, batch.edge_attr, batch.edge_type
        indices = x[:, -1].clone().long()

        for conv, bn in zip(self.convs, self.bns):
            if self.use_edge_attr:
                x = conv(x, edge_index, edge_type=edge_type, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index, edge_type=edge_type)
            x = F.relu(x)
            if self.training:
                x = self.dropout_layer(x)
            if self.norm == 'bn':
                x = bn(x)
            elif self.norm == 'ln':
                x = bn(x, batch=batch.batch)

        # Pooling
        x_prot = self.pooling_layer(x[indices == 0], batch=batch.batch[indices == 0])
        x_lig = self.pooling_layer(x[indices == 1], batch=batch.batch[indices == 1])

        # Concatenate protein and ligand representations
        out = torch.cat([x_prot, x_lig], dim=1)

        # Fully connected layers
        for layer in self.fcs[:-1]:
            out = layer(out)
            out = self.relu(out)
        # Output layer
        out = self.fcs[-1](out)

        return out

    def configure_optimizers(self):
        print("Learning rate: ", self.lr)
        # optimizer = Adam(self.parameters(), lr=self.lr)
        # return optimizer
        optimizer = Adam(self.parameters(), lr=self.lr)
        if self.use_ls:
            print(F"Using ExponentialLR. Gamma: {self.ex_ls_gamma}")
            scheduler = ExponentialLR(
                optimizer,
                gamma=self.ex_ls_gamma,  # Multiply LR by 0.95 every epoch
            )
            return [optimizer], [scheduler]
        return optimizer

    def training_step(self, batch, batch_idx):
        y_hat = self.forward(batch).view(-1)
        y = batch.y

        loss = F.mse_loss(y_hat, y)
        self.log("train/y_hat_-logKdKi", y_hat[0], batch_size=y.shape[0])
        self.log("train/y_-logKdKi", y[0], batch_size=y.shape[0])
        self.log("train_loss", loss, batch_size=y.shape[0], on_step=True, on_epoch=True, prog_bar=True, logger=True)

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

        # Calculate RMSE
        rmse = torch.sqrt(F.mse_loss(y_hats, ys))

        # Calculate MAE
        mae = F.l1_loss(y_hats, ys)

        # Log the metrics
        self.log(f"{prefix}/pearson_corr", pearson_corr, on_epoch=True)
        self.log(f"{prefix}/spearman_corr", spearman_corr, on_epoch=True)
        self.log(f"{prefix}/rmse", rmse, on_epoch=True)
        self.log(f"{prefix}/mae", mae, on_epoch=True)

    def on_validation_epoch_end(self):
        y_hats, ys = zip(*self.validation_step_outputs)
        self.log_metrics(y_hats, ys, "val")
        self.validation_step_outputs.clear()
    
    def on_test_epoch_end(self):
        if len(self.test_step_outputs) > 0:
            y_hats, ys = zip(*self.test_step_outputs)
            self.log_metrics(y_hats, ys, "test")
            self.test_step_outputs.clear()

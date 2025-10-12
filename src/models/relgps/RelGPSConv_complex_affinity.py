import torch
from pytorch_lightning import LightningModule
from torch.nn import Linear, ReLU, ModuleList, Sequential
from torch.nn import functional as F
from torch.optim import Adam
from torch_geometric.nn import GPSConv, RGCNConv, global_mean_pool, global_add_pool, global_max_pool, RGATConv
from torch_geometric.nn.attention import PerformerAttention
from scipy.stats import pearsonr, spearmanr
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR

class RedrawProjection:
    def __init__(self, model, redraw_interval=None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1

class GPSConv_complex_affinity(LightningModule):
    """
    Graph Transformer model for predicting protein-ligand binding affinity.
    Uses GPSConv with RGCN as local message passing layer.
    """

    def __init__(self, in_dim, out_dim=1, hidden_dim=64, lr=1e-4, num_relations=2,
                 num_gnn_layers=5, num_fc_layers=2, dropout=0.1, heads=4,
                 pooling='mean', attn_type='multihead', norm='batch_norm',
                 use_ls=False, ex_ls_gamma=1.0, log_every_step=False, local_gnn_type='rgcn'):
        super(GPSConv_complex_affinity, self).__init__()
        self.save_hyperparameters()
        self.local_gnn_type = local_gnn_type
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        self.num_relations = num_relations
        self.lr = lr
        self.dropout = dropout
        self.pooling = pooling
        self.use_ls = use_ls
        self.ex_ls_gamma = ex_ls_gamma
        self.log_every_step = log_every_step
        self.attn_type = attn_type
        
        # Initial node embedding
        self.embedding = Linear(in_dim, hidden_dim)
        
        # GPS layers
        self.convs = ModuleList()
        attn_kwargs = {'dropout': dropout}
        
        for _ in range(num_gnn_layers):
            # Create RGCN as local message passing layer
            # 根据 local_gnn_type 选择使用 RGCN 或 RGAT
            if self.local_gnn_type == 'rgat':
                local_mp = RGATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    num_relations=num_relations
                )
            else:  # 默认使用 RGCN
                local_mp = RGCNConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    num_relations=num_relations
                )
            # rgcn = RGCNConv(
            #     in_channels=hidden_dim,
            #     out_channels=hidden_dim,
            #     num_relations=num_relations
            # )
            
            # Create GPSConv with RGCN
            conv = GPSConv(
                channels=hidden_dim,
                conv=local_mp,
                heads=heads,
                dropout=dropout,
                attn_type=attn_type,
                attn_kwargs=attn_kwargs,
                norm=norm
            )
            self.convs.append(conv)
        
        # Fully connected layers
        self.fcs = ModuleList()
        fc_input_dim = hidden_dim * 2  # Concatenated protein and ligand representations
        
        for _ in range(num_fc_layers - 1):
            self.fcs.append(Linear(fc_input_dim, fc_input_dim))
            self.fcs.append(ReLU())
        
        # Output layer
        self.fcs.append(Linear(fc_input_dim, out_dim))
        
        # Pooling function
        if self.pooling == "mean":
            self.pooling_layer = global_mean_pool
        elif self.pooling == "add":
            self.pooling_layer = global_add_pool
        elif self.pooling == "max":
            self.pooling_layer = global_max_pool
        
        # Setup redraw projection for performer attention
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None
        )
        
        print(f"Model initialized with {num_gnn_layers} GPS layers, {num_fc_layers} FC layers")
        print(f"Attention type: {attn_type}, Local GNN: {local_gnn_type}, Pooling: {pooling}, Normalization: {norm}")

    def forward(self, batch):
        x, edge_index, edge_type = batch.x, batch.edge_index, batch.edge_type
        
        # Get protein/ligand indicator (last feature dimension)
        indices = x[:, -1].clone().long()
        
        # Initial embedding
        x = self.embedding(x)
        
        # Process through GPS layers
        for conv in self.convs:
            # Pass both edge_index and edge_type to GPSConv
            # edge_type will be forwarded to the internal RGCN
            x = conv(x, edge_index, batch=batch.batch, edge_type=edge_type)
        
        # Pooling for protein and ligand separately
        x_prot = self.pooling_layer(x[indices == 0], batch=batch.batch[indices == 0])
        x_lig = self.pooling_layer(x[indices == 1], batch=batch.batch[indices == 1])
        
        # Concatenate protein and ligand representations
        out = torch.cat([x_prot, x_lig], dim=1)
        
        # FC layers
        for layer in self.fcs:
            out = layer(out)
        
        return out

    def configure_optimizers(self):
        print("Learning rate:", self.lr)
        optimizer = Adam(self.parameters(), lr=self.lr)
        
        if self.use_ls:
            print(f"Using ExponentialLR. Gamma: {self.ex_ls_gamma}")
            scheduler = ExponentialLR(optimizer, gamma=self.ex_ls_gamma)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return optimizer

    def training_step(self, batch, batch_idx):
        # Redraw projection matrices if using performer attention
        self.redraw_projection.redraw_projections()
        
        y_hat = self.forward(batch).view(-1)
        y = batch.y
        
        loss = F.mse_loss(y_hat, y)
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
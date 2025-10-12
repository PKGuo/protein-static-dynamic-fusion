import torch
from pytorch_lightning import LightningModule
from torch.nn import Linear, ReLU, BatchNorm1d
from torch_geometric.nn import RGCNConv
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from scipy.stats import pearsonr, spearmanr
from torch.nn import functional as F

class RGCNNodeRegressor(LightningModule):
    """
    Relational Graph Convolutional Network for node regression.
    """

    def __init__(self, in_dim, hidden_dim=64, num_relations=2, lr=1e-4,
                 num_gnn_layers=5, num_fc_layers=2, root_weight=True):
        super(RGCNNodeRegressor, self).__init__()
        self.save_hyperparameters()
        self.num_relations = num_relations
        self.lr = lr

        # GNN layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # Input layer
        self.convs.append(RGCNConv(in_dim, hidden_dim, num_relations=num_relations, root_weight=root_weight))
        self.bns.append(BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_gnn_layers - 1):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations, root_weight=root_weight))
            self.bns.append(BatchNorm1d(hidden_dim))

        # FC layers
        self.fcs = torch.nn.ModuleList()
        for _ in range(num_fc_layers - 1):
            self.fcs.append(Linear(hidden_dim, hidden_dim))
            self.fcs.append(ReLU())
        # Output layer
        self.fcs.append(Linear(hidden_dim, 1))  # 单个输出

        self.loss_fn = torch.nn.MSELoss()  # 回归损失

        # Metrics
        self.mae = MeanAbsoluteError()
        self.rmse = MeanSquaredError(squared=False)  # RMSE

        # Initialize lists to store predictions and ground truth
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        print("The model RGCNNodeRegressor is initialized!")

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        edge_type = batch.edge_type

        # GNN layers
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = bn(x)

        # FC layers
        for layer in self.fcs[:-1]:
            x = layer(x)
        out = self.fcs[-1](x)

        return out

    def training_step(self, batch, batch_idx):
        y_hat = self.forward(batch).squeeze()
        y = batch.y

        loss = self.loss_fn(y_hat, y)
        
        self.train_step_outputs.append((y_hat.detach().cpu(), y.detach().cpu()))

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=y.shape[0])

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.forward(batch).squeeze()
        y = batch.y

        loss = self.loss_fn(y_hat, y)

        self.log("val/loss", loss, batch_size=y.shape[0], add_dataloader_idx=False)
        self.validation_step_outputs.append((y_hat.detach().cpu(), y.detach().cpu()))
            
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

        # Convert to numpy arrays to calculate correlation coefficients
        y_hats_np = y_hats.numpy()
        ys_np = ys.numpy()

        # Calculate Pearson and Spearman correlation coefficients
        pearson_corr, _ = pearsonr(y_hats_np, ys_np)
        spearman_corr, _ = spearmanr(y_hats_np, ys_np)

        # Calculate RMSE and MAE
        rmse = torch.sqrt(F.mse_loss(y_hats, ys))
        mae = F.l1_loss(y_hats, ys)

        self.log(f"{prefix}/pearson_corr", pearson_corr, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/spearman_corr", spearman_corr, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/rmse", rmse, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/mae", mae, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self):
        if len(self.train_step_outputs) > 0:
            y_hats, ys = zip(*self.train_step_outputs)
            y_hats = torch.cat(y_hats)
            ys = torch.cat(ys)

            # Convert to numpy arrays to calculate correlation coefficients
            y_hats_np = y_hats.numpy()
            ys_np = ys.numpy()

            # Calculate Pearson and Spearman correlation coefficients
            pearson_corr, _ = pearsonr(y_hats_np, ys_np)
            # spearman_corr, _ = spearmanr(y_hats_np, ys_np)

            # Calculate RMSE and MAE
            rmse = torch.sqrt(F.mse_loss(y_hats, ys))
            mae = F.l1_loss(y_hats, ys)

            self.log("train_epoch/pearson_corr", pearson_corr, on_epoch=True, prog_bar=True)
            # self.log("train_epoch/spearman_corr", spearman_corr, on_epoch=True, prog_bar=True)
            self.log("train_epoch/rmse", rmse, on_epoch=True, prog_bar=True)
            self.log("train_epoch/mae", mae, on_epoch=True, prog_bar=True)

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

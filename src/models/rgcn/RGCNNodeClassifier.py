import torch
from pytorch_lightning import LightningModule
from torch.nn import Linear, ReLU, BatchNorm1d, CrossEntropyLoss
from torch_geometric.nn import RGCNConv
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC, BinaryAveragePrecision
from torch.nn import functional as F

class RGCNNodeClassifier(LightningModule):
    """
    Relational Graph Convolutional Network for node classification.
    """

    def __init__(self, in_dim, hidden_dim=64, num_relations=2, num_classes=2, lr=1e-3,
                 num_gnn_layers=5, num_fc_layers=2, root_weight=True, use_class_weights=False):
        super(RGCNNodeClassifier, self).__init__()
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
        self.fcs.append(Linear(hidden_dim, num_classes))
        self.use_class_weights = use_class_weights
        if use_class_weights:
            class_weights = torch.tensor([0.1279, 0.8721])
            self.loss_fn = CrossEntropyLoss(weight=class_weights)
            print('------------------------------')
            print(f"Using class weights: {class_weights}") 
        else:
            print('------------------------------')
            print(f"No Using class weights") 
            self.loss_fn = CrossEntropyLoss()

        # Metrics
        self.acc = BinaryAccuracy()
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()
        self.f1 = BinaryF1Score()
        # New binary-specific metrics
        self.aucroc = BinaryAUROC()
        self.aucpr = BinaryAveragePrecision()

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
        logits = self.forward(batch)
        y = batch.y
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        
        acc = self.acc(preds, y)
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)
        f1 = self.f1(preds, y)

        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=y.shape[0])
        self.log('train/acc', acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=y.shape[0])
        self.log('train/precision', precision, on_step=True, on_epoch=True, prog_bar=True, batch_size=y.shape[0])
        self.log('train/recall', recall, on_step=True, on_epoch=True, prog_bar=True, batch_size=y.shape[0])
        self.log('train/f1', f1, on_step=True, on_epoch=True, prog_bar=True, batch_size=y.shape[0])


        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)
        y = batch.y
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)[:, 1]

        # Validation set
        acc = self.acc(preds, y)
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)
        f1 = self.f1(preds, y)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True, add_dataloader_idx=False, batch_size=y.shape[0])
        self.log("val/acc", acc, on_epoch=True, prog_bar=True, add_dataloader_idx=False, batch_size=y.shape[0])
        self.log("val/precision", precision, on_epoch=True, prog_bar=True, add_dataloader_idx=False, batch_size=y.shape[0])
        self.log("val/recall", recall, on_epoch=True, prog_bar=True, add_dataloader_idx=False, batch_size=y.shape[0])
        self.log("val/f1", f1, on_epoch=True, prog_bar=True, add_dataloader_idx=False, batch_size=y.shape[0])
                # Add new metrics to validation logging
        self.log("val/aucroc", self.aucroc(probs, y), on_epoch=True, prog_bar=True, 
                add_dataloader_idx=False, batch_size=y.shape[0])
        self.log("val/aucpr", self.aucpr(probs, y), on_epoch=True, prog_bar=True, 
                add_dataloader_idx=False, batch_size=y.shape[0])

        return loss

    def test_step(self, batch, batch_idx):
        logits = self.forward(batch)
        y = batch.y
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.acc(preds, y)
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)
        f1 = self.f1(preds, y)

        probs = F.softmax(logits, dim=1)[:, 1]

        self.log('test/loss', loss, on_epoch=True, prog_bar=True, batch_size=y.shape[0])
        self.log('test/acc', acc, on_epoch=True, prog_bar=True, batch_size=y.shape[0])
        self.log('test/precision', precision, on_epoch=True, prog_bar=True, batch_size=y.shape[0])
        self.log('test/recall', recall, on_epoch=True, prog_bar=True, batch_size=y.shape[0])
        self.log('test/f1', f1, on_epoch=True, prog_bar=True, batch_size=y.shape[0])

        self.log("test/aucroc", self.aucroc(probs, y), on_epoch=True, prog_bar=True, batch_size=y.shape[0])
        self.log("test/aucpr", self.aucpr(probs, y), on_epoch=True, prog_bar=True, batch_size=y.shape[0])

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
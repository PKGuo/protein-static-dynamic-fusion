import torch
from pytorch_lightning import LightningModule
from torch.nn import Linear, ReLU, BatchNorm1d, CrossEntropyLoss
from torch_geometric.nn import RGATConv
from torch_geometric.nn.norm import BatchNorm, LayerNorm
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC, BinaryAveragePrecision
from torch.nn import functional as F
class RGATNodeClassifier(LightningModule):
    """
    Relational Graph Convolutional Network for node classification.
    """

    def __init__(self, in_dim, hidden_dim=64, num_relations=2, num_classes=2, lr=1e-3,
                  num_gnn_layers=5, num_fc_layers=2, heads=1, edge_attr_dim=1, concat=False,
             use_edge_attr=True, gnn_dropout = 0.0, dropout=0.1, norm='bn', use_class_weights=False):
        super(RGATNodeClassifier, self).__init__()
        self.save_hyperparameters()
        self.num_relations = num_relations
        self.lr = lr
        self.norm = norm
        # GNN layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.gnn_dropout = gnn_dropout
        self.dropout = dropout
        self.use_edge_attr = use_edge_attr

        if self.use_edge_attr:
            edge_dim = edge_attr_dim  # Set edge_dim to edge_attr dimensionality
        else:
            edge_dim = None  # Don't use edge_attr, edge_dim is None

        # Calculate the effective hidden dimension considering heads and concat
        effective_hidden_dim = hidden_dim * heads if concat else hidden_dim

        self.convs.append(RGATConv(
            in_dim, hidden_dim, num_relations=num_relations, heads=heads,
            edge_dim=edge_dim, concat=concat, dropout=self.gnn_dropout
        ))
        if self.norm == 'bn':
            self.bns.append(BatchNorm(effective_hidden_dim))
        elif self.norm == 'ln':
            self.bns.append(LayerNorm)
 
        for _ in range(num_gnn_layers - 1):
            self.convs.append(RGATConv(
                effective_hidden_dim, hidden_dim, num_relations=num_relations, heads=heads,
                edge_dim=edge_dim, concat=concat, dropout=self.gnn_dropout
            ))
            if norm == 'bn':
                self.bns.append(BatchNorm(effective_hidden_dim))
            elif norm == 'ln':
                self.bns.append(LayerNorm(effective_hidden_dim))

        # FC layers
        self.fcs = torch.nn.ModuleList()
        for _ in range(num_fc_layers - 1):
            self.fcs.append(Linear(hidden_dim, hidden_dim))

        # Output layer
        self.fcs.append(Linear(hidden_dim, num_classes))

        self.dropout_layer = torch.nn.Dropout(p=self.dropout)
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

        x, edge_index, edge_attr, edge_type = batch.x, batch.edge_index, batch.edge_attr, batch.edge_type

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


        # FC layers
        for layer in self.fcs[:-1]:
            x = layer(x)         
            x = F.relu(x)
            if self.training:
                x = self.dropout_layer(x)

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
    
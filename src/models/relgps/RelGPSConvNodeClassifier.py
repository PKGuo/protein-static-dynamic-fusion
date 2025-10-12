import torch
from pytorch_lightning import LightningModule
from torch.nn import Linear, ReLU, CrossEntropyLoss, ModuleList
from torch.nn import functional as F
from torch_geometric.nn import GPSConv, RGCNConv, RGATConv
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC, BinaryAveragePrecision
from torch_geometric.nn.attention import PerformerAttention

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

class GPSConvNodeClassifier(LightningModule):
    """
    Graph Transformer model for node classification.
    Uses GPSConv with RGCN as local message passing layer.
    """

    def __init__(self, in_dim, hidden_dim=64, out_dim=2, num_relations=2, lr=1e-3,
                 num_gnn_layers=5, num_fc_layers=2, dropout=0.1, heads=4,
                 attn_type='multihead', norm='batch_norm', use_class_weights=False,
                 log_every_step=False, local_gnn_type='rgcn'):
        super(GPSConvNodeClassifier, self).__init__()
        self.save_hyperparameters()
        self.local_gnn_type = local_gnn_type
        self.num_relations = num_relations
        self.lr = lr
        self.dropout = dropout
        self.log_every_step = log_every_step
        self.attn_type = attn_type
        
        # Initial node embedding
        self.embedding = Linear(in_dim, hidden_dim)
        
        # GPS layers
        self.convs = ModuleList()
        attn_kwargs = {'dropout': dropout}
        
        for _ in range(num_gnn_layers):
            # Create RGCN as local message passing layer
            # rgcn = RGCNConv(
            #     in_channels=hidden_dim,
            #     out_channels=hidden_dim,
            #     num_relations=num_relations
            # )
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
        
        # FC layers
        self.fcs = ModuleList()
        for _ in range(num_fc_layers - 1):
            self.fcs.append(Linear(hidden_dim, hidden_dim))
            self.fcs.append(ReLU())
        
        # Output layer
        self.fcs.append(Linear(hidden_dim, out_dim))
        
        # Dropout layer
        self.dropout_layer = torch.nn.Dropout(p=dropout)
        
        # Loss function with optional class weights
        self.use_class_weights = use_class_weights
        if use_class_weights:
            # Register as buffer so it moves with model to correct device
            self.register_buffer('class_weights', torch.tensor([0.1279, 0.8721]))
            self.loss_fn = CrossEntropyLoss(weight=self.class_weights)
            print('------------------------------')
            print(f"Using class weights: {self.class_weights}")
        else:
            print('------------------------------')
            print(f"Not using class weights")
            self.loss_fn = CrossEntropyLoss()
        
        # Setup redraw projection for performer attention
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None
        )
        
        # Metrics
        self.acc = BinaryAccuracy()
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()
        self.f1 = BinaryF1Score()
        self.aucroc = BinaryAUROC()
        self.aucpr = BinaryAveragePrecision()
        
        print("Learning rate:", self.lr)
        print(f"Model initialized with {num_gnn_layers} GPS layers, {num_fc_layers} FC layers")
        print(f"Attention type: {attn_type}, Local GNN: {local_gnn_type}, Normalization: {norm}")

    def forward(self, batch):
        x, edge_index, edge_type = batch.x, batch.edge_index, batch.edge_type
        
        # Initial embedding
        x = self.embedding(x)
        
        # Process through GPS layers
        for conv in self.convs:
            # Pass both edge_index and edge_type to GPSConv
            x = conv(x, edge_index, batch=batch.batch, edge_type=edge_type)
        
        # FC layers
        for i, layer in enumerate(self.fcs[:-1]):
            x = layer(x)
            if i % 2 == 1:  # Apply ReLU and dropout after each Linear+ReLU pair
                if self.training:
                    x = self.dropout_layer(x)
        
        # Output layer
        out = self.fcs[-1](x)
        return out

    def training_step(self, batch, batch_idx):
        # Redraw projection matrices if using performer attention
        self.redraw_projection.redraw_projections()
        
        logits = self.forward(batch)
        y = batch.y
        loss = self.loss_fn(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        
        acc = self.acc(preds, y)
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)
        f1 = self.f1(preds, y)
        
        # Log with step control
        self.log('train/loss', loss, on_step=self.log_every_step, on_epoch=True, prog_bar=True, batch_size=y.shape[0])
        self.log('train/acc', acc, on_step=self.log_every_step, on_epoch=True, prog_bar=True, batch_size=y.shape[0])
        self.log('train/precision', precision, on_step=self.log_every_step, on_epoch=True, prog_bar=True, batch_size=y.shape[0])
        self.log('train/recall', recall, on_step=self.log_every_step, on_epoch=True, prog_bar=True, batch_size=y.shape[0])
        self.log('train/f1', f1, on_step=self.log_every_step, on_epoch=True, prog_bar=True, batch_size=y.shape[0])
        
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)
        y = batch.y
        loss = self.loss_fn(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)[:, 1]
        
        acc = self.acc(preds, y)
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)
        f1 = self.f1(preds, y)
        
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, add_dataloader_idx=False, batch_size=y.shape[0])
        self.log("val/acc", acc, on_epoch=True, prog_bar=True, add_dataloader_idx=False, batch_size=y.shape[0])
        self.log("val/precision", precision, on_epoch=True, prog_bar=True, add_dataloader_idx=False, batch_size=y.shape[0])
        self.log("val/recall", recall, on_epoch=True, prog_bar=True, add_dataloader_idx=False, batch_size=y.shape[0])
        self.log("val/f1", f1, on_epoch=True, prog_bar=True, add_dataloader_idx=False, batch_size=y.shape[0])
        self.log("val/aucroc", self.aucroc(probs, y), on_epoch=True, prog_bar=True, add_dataloader_idx=False, batch_size=y.shape[0])
        self.log("val/aucpr", self.aucpr(probs, y), on_epoch=True, prog_bar=True, add_dataloader_idx=False, batch_size=y.shape[0])
        
        return loss

    def test_step(self, batch, batch_idx):
        logits = self.forward(batch)
        y = batch.y
        loss = self.loss_fn(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)[:, 1]
        
        acc = self.acc(preds, y)
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)
        f1 = self.f1(preds, y)
        
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
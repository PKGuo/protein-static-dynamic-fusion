import torch
from pytorch_lightning import LightningModule
from torch.nn import Linear, ReLU, CrossEntropyLoss
from torch.nn import functional as F
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC, BinaryAveragePrecision
from torch_geometric.nn.norm import BatchNorm, LayerNorm

from .regnn_ensemble import REGCLEnsemble

class REGNNNodeClassifier(LightningModule):
    """
    Equivariant Relational Graph Neural Network for node classification.
    Uses REGCLEnsemble to maintain equivariance while handling multiple relation types.
    """

    def __init__(self, in_dim, hidden_dim=64, out_dim=2, num_relations=2, lr=1e-3,
                 num_gnn_layers=5, num_fc_layers=2, dropout=0.1, gnn_dropout=0.0,
                 node_agg='sum', coords_agg='mean', attention=False, normalize=False,
                 tanh=False, use_edge_attr=True, edge_attr_dim=1, use_class_weights=False,
                 norm='bn', residual=True, log_every_step=False, update_coords=False):
        super(REGNNNodeClassifier, self).__init__()
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
        
        # FC layers
        self.fcs = torch.nn.ModuleList()
        for i in range(num_fc_layers - 1):
            self.fcs.append(Linear(hidden_dim, hidden_dim))
        
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
        
        # Metrics
        self.acc = BinaryAccuracy()
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()
        self.f1 = BinaryF1Score()
        self.aucroc = BinaryAUROC()
        self.aucpr = BinaryAveragePrecision()
        
    def forward(self, batch):
        x, pos = batch.x, batch.pos
        edge_index = batch.edge_index
        edge_type = batch.edge_type
        edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') and self.use_edge_attr else None
        
        # Apply embedding to input features
        x = self.embedding(x)
        
        # Process through GNN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h, pos = conv(x, edge_index, pos, edge_attr, edge_type)
            
            # Check for NaNs
            if torch.isnan(h).any():
                print(f"NaN detected after conv layer {i}")
                
                # Find which nodes have NaNs
                nan_nodes = torch.where(torch.isnan(h).any(dim=1))[0]
                print(f"Nodes with NaNs: {nan_nodes.tolist()}")
                
                # Find which graphs these nodes belong to
                if hasattr(batch, 'batch'):
                    problem_graphs = torch.unique(batch.batch[nan_nodes]).tolist()
                    print(f"Graphs with NaNs: {problem_graphs}")
                    
                    # If you have a batch.pdb_id list, you can print those too
                    if hasattr(batch, 'pdb_id') and len(batch.pdb_id) == len(torch.unique(batch.batch)):
                        problem_pdbs = [batch.pdb_id[i] for i in problem_graphs]
                        print(f"PDB IDs with NaNs: {problem_pdbs}")
                
                # Return dummy tensor to prevent further NaNs
                return torch.zeros((x.size(0), 2), device=x.device)  # Hard-coded out_dim=2
            
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
        
        # Check before final layer
        if torch.isnan(x).any():
            print("NaN detected before final layer")
            nan_nodes = torch.where(torch.isnan(x).any(dim=1))[0]
            print(f"Nodes with NaNs: {nan_nodes.tolist()}")
            if hasattr(batch, 'batch'):
                problem_graphs = torch.unique(batch.batch[nan_nodes]).tolist()
                print(f"Graphs with NaNs: {problem_graphs}")
            
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
        
        # Log with step control
        # self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=y.shape[0])
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
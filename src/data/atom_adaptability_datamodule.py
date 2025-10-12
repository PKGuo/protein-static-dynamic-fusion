import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from src.data.datasets.atom_adaptability_dataset import AtomAdaptabilityGraphDataset
import os

class AtomAdaptabilityDataModule(pl.LightningDataModule):
    def __init__(
        self,
        graph_h5_path,
        pdb_list_train_path,
        pdb_list_val_path,
        pdb_list_test_path,
        graph_type="distance",
        batch_size=32,
        num_workers=1,
    ):
        super().__init__()
        self.graph_h5_path = graph_h5_path
        self.pdb_list_train_path = pdb_list_train_path
        self.pdb_list_val_path = pdb_list_val_path
        self.pdb_list_test_path = pdb_list_test_path

        # 检查文件路径是否存在
        assert os.path.exists(graph_h5_path), f"Graph data path not found: {graph_h5_path}"
        assert os.path.exists(pdb_list_train_path), f"Training PDB list not found: {pdb_list_train_path}"
        assert os.path.exists(pdb_list_val_path), f"Validation PDB list not found: {pdb_list_val_path}"
        assert os.path.exists(pdb_list_test_path), f"Test PDB list not found: {pdb_list_test_path}"

        # Load PDB lists from files
        with open(pdb_list_train_path, "r") as f:
            self.pdb_list_train = [line.strip() for line in f]
        with open(pdb_list_val_path, "r") as f:
            self.pdb_list_val = [line.strip() for line in f]
        with open(pdb_list_test_path, "r") as f:
            self.pdb_list_test = [line.strip() for line in f]

        self.graph_type = graph_type
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.dataset_train = AtomAdaptabilityGraphDataset(
                self.graph_h5_path, self.pdb_list_train, graph_type=self.graph_type
            )
            self.dataset_val = AtomAdaptabilityGraphDataset(
                self.graph_h5_path, self.pdb_list_val, graph_type=self.graph_type
            )
            self.dataset_test = AtomAdaptabilityGraphDataset(
                self.graph_h5_path, self.pdb_list_test, graph_type=self.graph_type
            )

        if stage == 'test' or stage is None:
            self.dataset_test = AtomAdaptabilityGraphDataset(
                self.graph_h5_path, self.pdb_list_test, graph_type=self.graph_type
            )
            
    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        assert len(self.dataset_val) > 0, "Validation set is empty!"
        return DataLoader(
            self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

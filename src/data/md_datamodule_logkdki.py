import os
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from src.data.datasets.md_dataset_logkdki import LogKdKiDataset  # 修改为新的数据集

class LogKdKiDataModule(pl.LightningDataModule):
    def __init__(
        self,
        graph_path,
        pdb_list_train,
        pdb_list_test,
        pdb_list_val,
        graph_type="distance",
        misato_features=False,
        batch_size=32,
        num_workers=16,
    ):
        """
        A data module for the LogKdKiDataset.

        Parameters:
        - graph_path: Path to the preprocessed graph data.
        - pdb_list_train: Path to the pickle file with PDB codes for training data.
        - pdb_list_test: Path to the pickle file with PDB codes for test data.
        - pdb_list_val: Path to the pickle file with PDB codes for validation data.
        - graph_type: Type of graph to load. Defaults to "distance".
        - misato_features: Whether to add MISATO features. Defaults to False.
        - batch_size: Batch size for the data loaders.
        - num_workers: Number of workers for the data loaders.
        """
        super().__init__()
        assert os.path.exists(graph_path), f"Graph data path not found: {graph_path}"
        assert os.path.exists(pdb_list_train), f"Training PDB list not found: {pdb_list_train}"
        assert os.path.exists(pdb_list_test), f"Test PDB list not found: {pdb_list_test}"
        assert os.path.exists(pdb_list_val), f"Validation PDB list not found: {pdb_list_val}"

        self.graph_path = graph_path
        self.pdb_list_train = pdb_list_train
        self.pdb_list_test = pdb_list_test
        self.pdb_list_val = pdb_list_val
        self.graph_type = graph_type
        self.misato_features = misato_features
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """Set up the data module based on the stage (fit/test)."""
        if stage == "fit" or stage is None:
            self.dataset_train = LogKdKiDataset(
                self.graph_path, self.pdb_list_train, graph_type=self.graph_type, misato_features=self.misato_features
            )
            self.dataset_val = LogKdKiDataset(
                self.graph_path, self.pdb_list_val, graph_type=self.graph_type, misato_features=self.misato_features
            )
            self.dataset_test = LogKdKiDataset(
                self.graph_path, self.pdb_list_test, graph_type=self.graph_type, misato_features=self.misato_features
            )

        if stage == "test" or stage is None:
            self.dataset_test = LogKdKiDataset(
                self.graph_path, self.pdb_list_test, graph_type=self.graph_type, misato_features=self.misato_features
            )

    def train_dataloader(self):
        """Returns the train data loader."""
        assert len(self.dataset_train) > 0, "Training set is empty!"
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        """Returns the validation data loader."""
        assert len(self.dataset_val) > 0, "Validation set is empty!"
        return DataLoader(
            self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False
        )

    def test_dataloader(self):
        """Returns the test data loader."""
        assert len(self.dataset_test) > 0, "Test set is empty!"
        return DataLoader(
            self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers
        )

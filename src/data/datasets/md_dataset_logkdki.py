import h5py
import torch as t
from torch_geometric.data import Dataset, Data
from src.utils.logutils import get_logger
import pickle

logger = get_logger(__name__)

def normalized(feature):
    feature = feature - feature.mean()
    feature = feature / feature.std()
    return feature

def load_entity_logkdki(entity_id, graph, graph_type="distance", misato_features=True):
    """
    Load entity data and calculate properties, using -logKd/Ki as the target.

    Parameters:
    - entity_id (str): ID of the entity to be loaded.
    - graph (h5py.File): HDF5 file containing graph data.
    - graph_type (str): Type of graph to load. Options are "distance", "correlation", "combined".
    - misato_features (bool, optional): Whether to add MISATO features.

    Returns:
    - Data: A torch_geometric.data.Data object containing the loaded data.
    """
    try:
        graph_entity = graph[entity_id]
    except KeyError:
        logger.error(f"Entity {entity_id} not found in graph file.")
        raise

    # Extracting features from the graph data
    coordinates = t.as_tensor(graph_entity["coordinates"][:]).to(t.float32)
    adaptabilities = t.as_tensor(graph_entity["adaptabilities"][:])
    charges = normalized(t.as_tensor(graph_entity["charges"][:]))
    x = t.as_tensor(graph_entity["atom_1hot"][:])
    lig_index = graph_entity["ligand_begin_index"][()]
    logkdki = t.as_tensor(graph_entity["-logKd_Ki"][()]).to(t.float32)
    
    # Determine edge_index and edge_attr based on graph_type
    if graph_type == "distance":
        edge_index = t.as_tensor(graph_entity["edge_index"][:])
        edge_attr = t.as_tensor(graph_entity["edge_attr"][:]).unsqueeze(1)
        edge_type = t.zeros(edge_index.shape[1], dtype=t.long)  # Single edge type
    elif graph_type == "correlation":
        edge_index = t.as_tensor(graph_entity["corr_edge_index"][:])
        edge_attr = t.as_tensor(graph_entity["corr_edge_attr"][:]).unsqueeze(1)
        edge_type = t.zeros(edge_index.shape[1], dtype=t.long)  # Single edge type
    elif graph_type == "combined":
        edge_index = t.as_tensor(graph_entity["combined_edge_index"][:])
        edge_attr = t.as_tensor(graph_entity["combined_edge_attr"][:]).unsqueeze(1)
        edge_type = t.as_tensor(graph_entity["combined_edge_type"][:])
    else:
        raise ValueError(f"Unsupported graph_type: {graph_type}")
    
    lig_len = int(charges.shape[0]) - lig_index
    nodes_numlig = t.cat((t.zeros(lig_index), t.ones(lig_len))).unsqueeze(1)
    
    if misato_features:
        x = t.cat((x, adaptabilities, charges, nodes_numlig), 1)
    else:
        x = t.cat((x, charges, nodes_numlig), 1)
    
    if graph_type == "combined":
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type, y=logkdki, pos=coordinates, pdb_id=entity_id)
    else:
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type, y=logkdki, pos=coordinates, pdb_id=entity_id)

class LogKdKiDataset(Dataset):
    def __init__(self, graph_path, pdb_list_path, graph_type="distance", misato_features=True):
        """
        Initialize the dataset for -logKd/Ki prediction.

        Parameters:
        - graph_path (str): Path to the HDF5 file containing graph data.
        - pdb_list_path (str): Path to the pickle file containing pdb codes.
        - graph_type (str, optional): Type of graph to load. Defaults to "distance".
        - misato_features (bool, optional): Whether to add MISATO features.
        """
        super().__init__(graph_path)
        self.graph = h5py.File(graph_path, "r")
        self.graph_type = graph_type
        self.misato_features = misato_features
        with open(pdb_list_path, "rb") as fp:
            self.pdb_codes = pickle.load(fp)

    def __len__(self):
        """Return length of the dataset."""
        return len(self.pdb_codes)

    def get(self, idx):
        """
        Get item at index idx.

        Parameters:
        - idx (int): Index of the item.

        Returns:
        - Data: A Data object representing the complex.
        """
        pdb_code = self.pdb_codes[idx]
        return load_entity_logkdki(pdb_code, self.graph, self.graph_type, self.misato_features)
    
    def __getitem__(self, idx: int):
        return self.get(idx)
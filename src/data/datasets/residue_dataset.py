import h5py
import torch
from torch_geometric.data import Data, Dataset

def load_entity_node_classification(entity_id, h5_file, graph_type="distance"):
    try:
        entity_group = h5_file[entity_id]
    except KeyError:
        print(f"Entity {entity_id} not found in the HDF5 file.")
        raise

    # Load node features and labels
    node_features = torch.tensor(entity_group['node_features'][:], dtype=torch.float)
    node_labels = torch.tensor(entity_group['node_labels'][:], dtype=torch.long)
    # Load coordinates (CA atom coordinates for residues)
    coordinates = torch.tensor(entity_group['ca_coords'][:], dtype=torch.float)

    # Load edge indices and edge attributes based on graph_type
    if graph_type == "distance":
        edge_index = torch.tensor(entity_group['edge_index_distance'][:], dtype=torch.long)
        edge_attr = torch.tensor(entity_group['edge_weight_distance'][:], dtype=torch.float).unsqueeze(1)
        edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long)  # Single edge type
    elif graph_type == "correlation":
        edge_index = torch.tensor(entity_group['edge_index_correlation'][:], dtype=torch.long)
        edge_attr = torch.tensor(entity_group['edge_weight_correlation'][:], dtype=torch.float).unsqueeze(1)
        edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long)  # Single edge type
    elif graph_type == "combined":
        edge_index = torch.tensor(entity_group['combined_edge_index'][:], dtype=torch.long)
        edge_attr = torch.tensor(entity_group['combined_edge_weight'][:], dtype=torch.float).unsqueeze(1)
        edge_type = torch.tensor(entity_group['combined_edge_type'][:], dtype=torch.long)
    else:
        raise ValueError(f"Unsupported graph_type: {graph_type}")

    # Create Data object
    if graph_type == "combined":
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_type=edge_type,
            y=node_labels,
            pos=coordinates,  # Add coordinates as pos
            pdb_id=entity_id
        )
    else:
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_type=edge_type,
            y=node_labels,
            pos=coordinates,  # Add coordinates as pos
            pdb_id=entity_id,
        )

    return data

class ResidueGraphDataset(Dataset):
    def __init__(self, h5_path, pdb_codes, graph_type="distance"):
        super().__init__()
        self.h5_path = h5_path
        self.pdb_codes = pdb_codes
        self.graph_type = graph_type
        self.h5_file = h5py.File(self.h5_path, 'r') # Will be opened in __getitem__
        print(">>> Opening HDF5 file during initialization...")

    def __len__(self):
        return len(self.pdb_codes)

    def get(self, idx):
        # if self.h5_file is None:
        #     # Each worker opens its own file handle
        #     self.h5_file = h5py.File(self.h5_path, 'r')
        #     print(">>> Opening HDF5 file...")
        pdb_code = self.pdb_codes[idx]
        data = load_entity_node_classification(pdb_code, self.h5_file, graph_type=self.graph_type)
        return data
    
    def get_item_by_pdb(self, pdb_code):
        # if self.h5_file is None:
        #     # Each worker opens its own file handle
        #     self.h5_file = h5py.File(self.h5_path, 'r')
        #     print(">>> Opening HDF5 file (get_item_by_pdb)...")
        data = load_entity_node_classification(pdb_code, self.h5_file, graph_type=self.graph_type)
        return data

    def __getitem__(self, idx):
        return self.get(idx)

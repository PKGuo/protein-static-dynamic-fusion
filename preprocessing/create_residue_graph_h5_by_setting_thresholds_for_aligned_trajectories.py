import h5py
import numpy as np
import os
from tqdm import tqdm
import torch
from scipy.spatial.distance import cdist
import pickle
import argparse

def load_mappings():
    # Load atoms_type_map to get the code corresponding to 'CX' (Cα atoms)
    with open('data/maps/atoms_type_map.pickle', 'rb') as f:
        typeMap = pickle.load(f)
    ca_type_codes = [code for code, type_str in typeMap.items() if type_str == 'CX']
    if not ca_type_codes:
        raise ValueError("Could not find the atoms_type code corresponding to 'CX'")
    ca_type_code = ca_type_codes[0]
    print(f"The atoms_type code corresponding to 'CX' is {ca_type_code}")

    # Load residue_Map to map residue codes to residue names
    with open('data/maps/atoms_residue_map.pickle', 'rb') as f:
        residue_Map = pickle.load(f)
    return ca_type_code, residue_Map

def get_ca_atoms(atoms_type, atoms_residue, atoms_coordinates, protein_indices, ca_type_code):
    protein_atoms_type = atoms_type[protein_indices]
    protein_atoms_residue = atoms_residue[protein_indices]
    protein_atoms_coordinates = atoms_coordinates[protein_indices]

    # Get Cα atom indices and coordinates
    ca_mask = protein_atoms_type == ca_type_code
    ca_indices = protein_indices[ca_mask]
    ca_coords = protein_atoms_coordinates[ca_mask]
    ca_residue_types = protein_atoms_residue[ca_mask]

    return ca_indices, ca_coords, ca_residue_types

def get_node_features(ca_residue_types, residue_Map):
    # Map residue codes to residue names
    residue_names = [residue_Map.get(res_type, 'UNK') for res_type in ca_residue_types]
    # Define allowable residues (20 standard amino acids + 'UNK')
    allowable_residues = [
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
        'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
        'THR', 'TRP', 'TYR', 'VAL', 'UNK'
    ]
    residue_to_idx = {res: idx for idx, res in enumerate(allowable_residues)}
    # Build node features (one-hot encoding)
    node_features = []
    for res_name in residue_names:
        one_hot = [0] * len(allowable_residues)
        idx = residue_to_idx.get(res_name, residue_to_idx['UNK'])
        one_hot[idx] = 1
        node_features.append(one_hot)
    node_features = np.array(node_features, dtype=np.float32)
    return node_features

def build_distance_graph(ca_coords, distance_threshold=10.0):
    # Compute CA-CA distance matrix
    distances = cdist(ca_coords, ca_coords)
    # Build distance-based graph
    edge_indices = np.argwhere(distances < distance_threshold)
    # Remove self-loops
    edge_indices = edge_indices[edge_indices[:, 0] != edge_indices[:, 1]]
    # Edge attributes: inverse distances
    edge_weights = 1.0 / (distances[edge_indices[:, 0], edge_indices[:, 1]] + 1e-6)  # Add small epsilon to avoid division by zero
    edge_weights = torch.tensor(edge_weights, dtype=torch.float32)
    # Convert to torch tensor and transpose to get edge_index shape [2, num_edges]
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    # print(edge_index[:, :5], edge_weights[:5])
    return edge_index, edge_weights

def build_correlation_graph(corr_mat, ca_indices, corr_threshold=0.5):
    # Extract CA-CA correlations
    ca_corr_mat = corr_mat[np.ix_(ca_indices, ca_indices)]
    # print("ca_corr_mat.shape",ca_corr_mat.shape)
    # Threshold the correlation matrix to get edges
    corr_edge_indices = np.argwhere(np.abs(ca_corr_mat) >= corr_threshold)
    # Remove self-loops
    corr_edge_indices = corr_edge_indices[corr_edge_indices[:, 0] != corr_edge_indices[:, 1]]
    # Edge attributes: correlation values 
    # In the future, we can try negetive threshold to get anti-correlation edges
    corr_edge_weights = ca_corr_mat[corr_edge_indices[:, 0], corr_edge_indices[:, 1]]
    corr_edge_weights = torch.tensor(corr_edge_weights, dtype=torch.float32)
    # Convert to torch tensor and transpose to get edge_index shape [2, num_edges]
    corr_edge_index = torch.tensor(corr_edge_indices, dtype=torch.long).t().contiguous()
    # print(corr_edge_index[:, :5], corr_edge_weights[:5])
    return corr_edge_index, corr_edge_weights

def process_pdb_code(pdb_code, group_in, group_out, ca_type_code, residue_Map, corr_matrix_dir, distance_threshold = 10.0, corr_threshold=0.5):
    # Read necessary data
    atoms_coordinates = group_in['atoms_coordinates_ref'][()]
    atoms_type = group_in['atoms_type'][()]
    atoms_residue_number = group_in['atoms_residue_number'][()]
    atoms_residue = group_in['atoms_residue'][()]
    residue_binding_labels = group_in['residue_binding_labels'][()]
    residue_ids = group_in['residue_ids'][()]
    molecules_begin_atom_index = group_in['molecules_begin_atom_index'][()]

    # Get total number of atoms
    total_atoms = len(atoms_type)

    # Identify ligand and protein atom indices
    molecules_begin_atom_index = molecules_begin_atom_index.astype(int)
    num_molecules = len(molecules_begin_atom_index)

    # Assume ligand is the last molecule
    ligand_start = molecules_begin_atom_index[-1]
    ligand_end = total_atoms
    ligand_indices = np.arange(ligand_start, ligand_end)

    # Protein atom indices (assuming protein is before the ligand)
    protein_end = ligand_start  # All atoms before the ligand
    protein_indices = np.arange(0, protein_end)

    # Get Cα atom indices, coordinates, and residue types
    ca_indices, ca_coords, ca_residue_types = get_ca_atoms(
        atoms_type, atoms_residue, atoms_coordinates, protein_indices, ca_type_code)

    # Build node features
    node_features = get_node_features(ca_residue_types, residue_Map)

    # Build node labels (binding site labels)
    residue_num_to_label = dict(zip(residue_ids, residue_binding_labels))
    # Map ca_residue_numbers to residue_ids
    ca_residue_numbers = atoms_residue_number[ca_indices]
    node_labels = [residue_num_to_label.get(res_num, 0) for res_num in ca_residue_numbers]
    node_labels = np.array(node_labels, dtype=np.int64)

    # Build distance graph
    # distance_threshold       # Adjust as needed
    edge_index_distance, edge_weight_distance = build_distance_graph(ca_coords, distance_threshold)

    # Build correlation graph
    corr_mat_path = os.path.join(corr_matrix_dir, f"{pdb_code}_correlation_data.npz")
    if os.path.exists(corr_mat_path):
        corr_data = np.load(corr_mat_path)
        corr_mat = corr_data['corr_all']
        # Build correlation graph
        # corr_threshold = 0.5  # Adjust as needed
        edge_index_correlation, edge_weight_correlation = build_correlation_graph(corr_mat, ca_indices, corr_threshold)
    else:
        print(f"Error loading correlation matrix for {pdb_code}")
        # raise an exception to stop running the code
        raise ValueError# Reraise the exception

    # Combine graphs
    combined_edge_index = torch.cat([edge_index_distance, edge_index_correlation], dim=1)
    combined_edge_weight = torch.cat([edge_weight_distance, edge_weight_correlation], dim=0)
    # Edge types: 0 for distance, 1 for correlation
    edge_type_distance = torch.zeros(edge_index_distance.size(1), dtype=torch.long)
    edge_type_correlation = torch.ones(edge_index_correlation.size(1), dtype=torch.long)
    combined_edge_type = torch.cat([edge_type_distance, edge_type_correlation], dim=0)

    # Save the graph data to the output HDF5 file
    grp = group_out
    grp.create_dataset('edge_index_distance', data=edge_index_distance.numpy())
    grp.create_dataset('edge_weight_distance', data=edge_weight_distance.numpy())
    grp.create_dataset('edge_index_correlation', data=edge_index_correlation.numpy())
    grp.create_dataset('edge_weight_correlation', data=edge_weight_correlation.numpy())
    grp.create_dataset('combined_edge_index', data=combined_edge_index.numpy())
    grp.create_dataset('combined_edge_weight', data=combined_edge_weight.numpy())
    grp.create_dataset('combined_edge_type', data=combined_edge_type.numpy())
    grp.create_dataset('node_features', data=node_features)
    grp.create_dataset('node_labels', data=node_labels)
    grp.create_dataset('ca_coords', data=ca_coords)
    # Save thresholds as attributes
    grp.attrs['distance_threshold'] = distance_threshold
    grp.attrs['correlation_threshold'] = corr_threshold


def main(args):
    # Paths to the input and output HDF5 files
    input_h5_path = 'data/processed/adaptability_MD_with_residue_numbers_labels.hdf5'

    # set threshold for distance and correlation
    distance_threshold = args.distance_threshold
    corr_threshold = args.corr_threshold

    # Output path with aligned indicator
    output_h5_path = f'data/processed/residue_graph_data_distance_{distance_threshold}_corr_aligned_{corr_threshold}.h5'
    
    # Use aligned correlation directory
    corr_matrix_dir = 'data/correlations/aligned_atomiccorr_no_hydrogen'

    # Load necessary mappings
    ca_type_code, residue_Map = load_mappings()

    # Open the input and output HDF5 files
    with h5py.File(input_h5_path, 'r') as h5_in, h5py.File(output_h5_path, 'w') as h5_out:
        pdb_codes = list(h5_in.keys())

        # Process only the first 5 data points for testing
        pdb_codes = pdb_codes

        for pdb_code in tqdm(pdb_codes, desc="Processing PDB codes"):
            group_in = h5_in[pdb_code]
            group_out = h5_out.create_group(pdb_code)

            try:
                process_pdb_code(pdb_code, group_in, group_out, ca_type_code, residue_Map, corr_matrix_dir, distance_threshold=distance_threshold, corr_threshold=corr_threshold)
            except Exception as e:
                print(f"Error processing {pdb_code}: {e}")

if __name__ == "__main__":
    # use parser to get the threshold values
    parser = argparse.ArgumentParser()
    parser.add_argument('--distance_threshold', type=float, default=10.0)
    parser.add_argument('--corr_threshold', type=float, default=0.3)#  # Changed default for aligned correlation. For unaligned, we use 0.5
    args = parser.parse_args()


    # use args as the arguments for the main function
    main(args)
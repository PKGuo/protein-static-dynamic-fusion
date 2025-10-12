import os
import h5py
import pickle
import numpy as np
import pandas as pd
import torch
import networkx as nx
from tqdm import tqdm
from scipy.spatial.distance import cdist

def get_node_features(atoms_element, element_map):
    allowable_elements = ['C', 'N', 'O', 'F', 'P', 'S', 'CL', 'BR', 'I', 'UNK']
    element_to_idx = {element: idx for idx, element in enumerate(allowable_elements)}
    element_numbers_to_symbols = {k: v.upper() for k, v in element_map.items()}

    node_features = []
    for elem_num in atoms_element:
        elem_symbol = element_numbers_to_symbols.get(elem_num, 'UNK')
        one_hot = [0] * len(allowable_elements)
        idx = element_to_idx.get(elem_symbol, element_to_idx['UNK'])
        one_hot[idx] = 1
        node_features.append(one_hot)
    return np.array(node_features, dtype=np.float32)

def build_distance_graph(coords, distance_threshold=4.5):
    distances = cdist(coords, coords)
    edge_indices = np.argwhere(distances < distance_threshold)
    edge_indices = edge_indices[edge_indices[:, 0] != edge_indices[:, 1]]
    edge_weights = 1.0 / (distances[edge_indices[:, 0], edge_indices[:, 1]] + 1e-6)
    edge_weights = torch.tensor(edge_weights, dtype=torch.float32)
    edge_index = torch.tensor(edge_indices.T, dtype=torch.long)
    return edge_index, edge_weights

def build_correlation_graph(corr_mat, corr_threshold=0.6):
    corr_edge_indices = np.argwhere(np.abs(corr_mat) >= corr_threshold)
    corr_edge_indices = corr_edge_indices[corr_edge_indices[:, 0] != corr_edge_indices[:, 1]]
    corr_edge_weights = corr_mat[corr_edge_indices[:, 0], corr_edge_indices[:, 1]]
    corr_edge_weights = torch.tensor(corr_edge_weights, dtype=torch.float32)
    corr_edge_index = torch.tensor(corr_edge_indices.T, dtype=torch.long)
    return corr_edge_index, corr_edge_weights

def process_pdb_code(pdb_code, group_in, group_out, element_map, corr_matrix_dir,
                     distance_threshold=4.5, corr_threshold=0.6):
    atoms_coordinates = group_in['atoms_coordinates_ref'][()]
    atoms_element = group_in['atoms_element'][()]
    feature_atoms_adaptability = group_in['feature_atoms_adaptability'][()]
    molecules_begin_atom_index = group_in['molecules_begin_atom_index'][()]

    molecules_begin_atom_index = molecules_begin_atom_index.astype(int)
    ligand_start = molecules_begin_atom_index[-1]
    protein_atoms_coordinates = atoms_coordinates[:ligand_start]
    protein_atoms_element = atoms_element[:ligand_start]
    protein_atoms_adaptability = feature_atoms_adaptability[:ligand_start]

    node_features = get_node_features(protein_atoms_element, element_map)
    node_labels = protein_atoms_adaptability.astype(np.float32)

    edge_index_distance, edge_weight_distance = build_distance_graph(protein_atoms_coordinates, distance_threshold)

    corr_mat_path = os.path.join(corr_matrix_dir, f"{pdb_code}_correlation_data.npz")
    if os.path.exists(corr_mat_path):
        corr_data = np.load(corr_mat_path)
        corr_mat = corr_data['corr_all']
        protein_corr_mat = corr_mat[:ligand_start, :ligand_start]
        edge_index_correlation, edge_weight_correlation = build_correlation_graph(protein_corr_mat, corr_threshold)
    else:
        print(f"Correlation matrix not found for {pdb_code}")
        raise ValueError(f"Correlation matrix not found for {pdb_code}")

    combined_edge_index = torch.cat([edge_index_distance, edge_index_correlation], dim=1)
    combined_edge_weight = torch.cat([edge_weight_distance, edge_weight_correlation], dim=0)
    edge_type_distance = torch.zeros(edge_index_distance.size(1), dtype=torch.long)
    edge_type_correlation = torch.ones(edge_index_correlation.size(1), dtype=torch.long)
    combined_edge_type = torch.cat([edge_type_distance, edge_type_correlation], dim=0)

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
    grp.create_dataset('atoms_coordinates', data=protein_atoms_coordinates)
    grp.create_dataset('atoms_element', data=protein_atoms_element)
    grp.attrs['distance_threshold'] = distance_threshold
    grp.attrs['correlation_threshold'] = corr_threshold

    if pdb_code in dataset_df.index:
        log_kd_ki = dataset_df.loc[pdb_code]['-logKd/Ki']
        grp.create_dataset('-logKd_Ki', data=log_kd_ki)
    else:
        print(f"Binding affinity not found for {pdb_code}")
        grp.create_dataset('-logKd_Ki', data=np.nan)

def main():
    distance_threshold = 4.5
    corr_threshold = 0.6  # Aligned correlation threshold

    input_h5_path = 'data/processed/adaptability_MD_with_residue_numbers_labels.hdf5'
    output_h5_path = f'data/processed/atom_graph_OnlyProtein_distance_{distance_threshold}_corr_aligned_{corr_threshold}.h5'

    # Use aligned correlation directory
    corr_matrix_dir = 'data/correlations/aligned_atomiccorr_no_hydrogen'

    element_map_path = 'data/maps/map_elements_numbers_number.pickle'
    with open(element_map_path, 'rb') as f:
        element_map = pickle.load(f)

    global dataset_df
    dataset_df = pd.read_csv("data/affinity_data.csv")
    dataset_df['PDB_code'] = dataset_df['PDB_code'].str.upper()
    dataset_df.set_index('PDB_code', inplace=True)

    with h5py.File(input_h5_path, 'r') as h5_in, h5py.File(output_h5_path, 'w') as h5_out:
        pdb_codes = list(h5_in.keys())
        for pdb_code in tqdm(pdb_codes, desc="Processing PDB codes"):
            group_in = h5_in[pdb_code]
            group_out = h5_out.create_group(pdb_code)
            try:
                process_pdb_code(pdb_code, group_in, group_out, element_map, corr_matrix_dir,
                                 distance_threshold=distance_threshold, corr_threshold=corr_threshold)
            except Exception as e:
                print(f"Error processing {pdb_code}: {e}")

if __name__ == '__main__':
    main()
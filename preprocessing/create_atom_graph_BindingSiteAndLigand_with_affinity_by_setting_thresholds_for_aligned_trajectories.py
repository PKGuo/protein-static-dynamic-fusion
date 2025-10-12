import h5py
import numpy as np
import os
from tqdm import tqdm
import torch
from scipy.spatial.distance import cdist
import pickle
import pandas as pd


def build_distance_graph(coords, distance_threshold=4.5):
    # Compute atom-atom distance matrix
    distances = cdist(coords, coords)
    # Build distance-based graph
    edge_indices = np.argwhere(distances < distance_threshold)
    # Remove self-loops
    edge_indices = edge_indices[edge_indices[:, 0] != edge_indices[:, 1]]
    # Edge attributes: inverse distances
    edge_weights = 1.0 / (distances[edge_indices[:, 0], edge_indices[:, 1]] + 1e-6)  # Avoid division by zero
    edge_weights = torch.tensor(edge_weights, dtype=torch.float32)
    # Convert to torch tensor and transpose to get edge_index shape [2, num_edges]
    edge_index = torch.tensor(edge_indices.T, dtype=torch.long)
    return edge_index, edge_weights

def build_correlation_graph(corr_mat, corr_threshold=0.73):
    # Threshold the correlation matrix to get edges
    corr_edge_indices = np.argwhere(np.abs(corr_mat) >= corr_threshold)
    # Remove self-loops
    corr_edge_indices = corr_edge_indices[corr_edge_indices[:, 0] != corr_edge_indices[:, 1]]
    # Edge attributes: correlation values
    corr_edge_weights = corr_mat[corr_edge_indices[:, 0], corr_edge_indices[:, 1]]
    corr_edge_weights = torch.tensor(corr_edge_weights, dtype=torch.float32)
    # Convert to torch tensor and transpose to get edge_index shape [2, num_edges]
    corr_edge_index = torch.tensor(corr_edge_indices.T, dtype=torch.long)
    return corr_edge_index, corr_edge_weights

def process_pdb_code(pdb_code, group_in, group_out, element_map, corr_matrix_dir,
                     preprocessed_h5, distance_threshold=4.5, corr_threshold=0.73):
    # Read necessary data from the input group
    atoms_coordinates = group_in['atoms_coordinates_ref'][()]
    atoms_residue_number = group_in['atoms_residue_number'][()]
    residue_binding_labels = group_in['residue_binding_labels'][()]
    residue_ids = group_in['residue_ids'][()]
    molecules_begin_atom_index = group_in['molecules_begin_atom_index'][()]
    atoms_element = group_in['atoms_element'][()]

    # Determine indices for ligand atoms
    molecules_begin_atom_index = molecules_begin_atom_index.astype(int)
    ligand_start = molecules_begin_atom_index[-1]
    ligand_atoms_indices = np.arange(ligand_start, len(atoms_coordinates))

    # Identify binding site residue indices
    binding_residue_indices = residue_ids[residue_binding_labels == 1]

    # Map atoms to residues
    atom_to_residue = atoms_residue_number

    # Identify binding site atom indices in protein
    binding_site_atom_indices = np.where(np.isin(atom_to_residue[:ligand_start], binding_residue_indices))[0]

    # Combine binding site atom indices and ligand atom indices
    selected_atom_indices = np.concatenate([binding_site_atom_indices, ligand_atoms_indices])

    # Extract selected atoms' data
    selected_atoms_coordinates = atoms_coordinates[selected_atom_indices]
    # Note: We will get node features from 'atom_1hot' in another file

    # Build node features from 'atom_1hot' in 'preprocessed_graph_invariant_combined.h5'
    if pdb_code in preprocessed_h5:
        atom_1hot = preprocessed_h5[pdb_code]['atom_1hot'][()]
        # Extract features for selected atoms
        node_features = atom_1hot[selected_atom_indices]


                # Extract charges
        charges = preprocessed_h5[pdb_code]['charges'][()]
        selected_atoms_charges = charges[selected_atom_indices]

                # Extract adaptabilities (if needed)
        adaptabilities = preprocessed_h5[pdb_code]['adaptabilities'][()]
        selected_atoms_adaptabilities = adaptabilities[selected_atom_indices]

    else:
        print(f"Atom features not found for {pdb_code} in preprocessed HDF5 file.")
        raise ValueError(f"Atom features not found for {pdb_code}")

    # Build distance graph
    edge_index_distance, edge_weight_distance = build_distance_graph(selected_atoms_coordinates, distance_threshold)

    # Build correlation graph
    corr_mat_path = os.path.join(corr_matrix_dir, f"{pdb_code}_correlation_data.npz")
    if os.path.exists(corr_mat_path):
        corr_data = np.load(corr_mat_path)
        corr_mat = corr_data['corr_all']
        # Get correlation matrix for selected atoms
        corr_mat_selected = corr_mat[np.ix_(selected_atom_indices, selected_atom_indices)]
        edge_index_correlation, edge_weight_correlation = build_correlation_graph(corr_mat_selected, corr_threshold)
    else:
        print(f"Correlation matrix not found for {pdb_code}")
        raise ValueError(f"Correlation matrix not found for {pdb_code}")

    # Combine graphs
    combined_edge_index = torch.cat([edge_index_distance, edge_index_correlation], dim=1)
    combined_edge_attr = torch.cat([edge_weight_distance, edge_weight_correlation], dim=0)
    # Edge types: 0 for distance, 1 for correlation
    edge_type_distance = torch.zeros(edge_index_distance.size(1), dtype=torch.long)
    edge_type_correlation = torch.ones(edge_index_correlation.size(1), dtype=torch.long)
    combined_edge_type = torch.cat([edge_type_distance, edge_type_correlation], dim=0)

    # Save the graph data to the output HDF5 file
    grp = group_out
    # Save node features
    grp.create_dataset('atom_1hot', data=node_features)
    # Save coordinates
    grp.create_dataset('coordinates', data=selected_atoms_coordinates)
        # Save charges
    grp.create_dataset('charges', data=selected_atoms_charges)
    # Save adaptabilities
    grp.create_dataset('adaptabilities', data=selected_atoms_adaptabilities)
    # Save edge indices and attributes
    grp.create_dataset('edge_index', data=edge_index_distance.numpy())
    grp.create_dataset('edge_attr', data=edge_weight_distance.numpy())
    grp.create_dataset('corr_edge_index', data=edge_index_correlation.numpy())
    grp.create_dataset('corr_edge_attr', data=edge_weight_correlation.numpy())
    grp.create_dataset('combined_edge_index', data=combined_edge_index.numpy())
    grp.create_dataset('combined_edge_attr', data=combined_edge_attr.numpy())
    grp.create_dataset('combined_edge_type', data=combined_edge_type.numpy())
    # Save ligand_begin_index
    ligand_begin_index = len(binding_site_atom_indices)
    grp.create_dataset('ligand_begin_index', data=ligand_begin_index)

    # Save binding affinity
    if pdb_code in dataset_df.index:
        affinity = dataset_df.loc[pdb_code]['-logKd/Ki']
        grp.create_dataset('affinity', data=affinity)
        grp.create_dataset('-logKd_Ki', data=affinity)
    else:
        print(f"Binding affinity not found for {pdb_code}")
        grp.create_dataset('affinity', data=np.nan)
        grp.create_dataset('-logKd_Ki', data=np.nan)

    # Save binding_site_atom_count
    binding_site_atom_count = len(binding_site_atom_indices)
    grp.create_dataset('binding_site_atom_count', data=binding_site_atom_count)

def main():
    # Set thresholds
    distance_threshold = 4.5
    corr_threshold = 0.6  # Adjusted as per your previous script

    # Paths to the input and output HDF5 files
    input_h5_path = 'data/processed/adaptability_MD_with_residue_numbers_labels.hdf5'
    preprocessed_h5_path = 'data/downloaded/preprocessed_graph_invariant_combined.h5'

    output_h5_path = f'data/processed/atom_graph_BindingSiteAndLigand_distance_{distance_threshold}_corr_aligned_{corr_threshold}.h5'
    corr_matrix_dir = 'data/correlations/aligned_atomiccorr_no_hydrogen'

    # Read the CSV file for binding affinity
    global dataset_df
    dataset_df = pd.read_csv("data/affinity_data.csv")
    dataset_df['PDB_code'] = dataset_df['PDB_code'].str.upper()
    dataset_df.set_index('PDB_code', inplace=True)

    # Open the input and output HDF5 files
    with h5py.File(input_h5_path, 'r') as h5_in, \
         h5py.File(preprocessed_h5_path, 'r') as preprocessed_h5, \
         h5py.File(output_h5_path, 'w') as h5_out:

        pdb_codes = list(h5_in.keys())
        preprocessed_pdb_codes = list(preprocessed_h5.keys())
        print(f"Number of PDB codes in input HDF5: {len(pdb_codes)}")
        print(f"Number of PDB codes in preprocessed HDF5: {len(preprocessed_pdb_codes)}")

        for pdb_code in tqdm(preprocessed_pdb_codes, desc="Processing PDB codes"):
            group_in = h5_in[pdb_code]
            group_out = h5_out.create_group(pdb_code)

            try:
                process_pdb_code(pdb_code, group_in, group_out, None, corr_matrix_dir,
                                 preprocessed_h5, distance_threshold=distance_threshold, corr_threshold=corr_threshold)
            except Exception as e:
                print(f"Error processing {pdb_code}: {e}")

if __name__ == "__main__":
    main()
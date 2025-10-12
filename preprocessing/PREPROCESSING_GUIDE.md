# Preprocessing Pipeline

This directory contains scripts for preprocessing molecular dynamics data and generating graph representations for protein-ligand binding analysis.

## Required Downloaded Files

Download these two files from MISATO dataset to `data/downloaded/`:

1. **adaptability_MD.hdf5** - MD adaptability and reference coordinates:
   ```bash
   wget -O data/downloaded/adaptability_MD.hdf5 https://syncandshare.lrz.de/getlink/fiCk9juiXYBKZ73VyHt372/adaptability_MD.hdf5
   ```

2. **preprocessed_graph_invariant_combined.h5** - Preprocessed invariant graphs:
   ```bash
   wget -O data/downloaded/preprocessed_graph_invariant_combined.h5 https://syncandshare.lrz.de/getlink/fi41DBaf6f1b6ZHSiGokU9/preprocessed_graph_invariant_numlig_affType.h5
   ```

For additional details on MISATO preprocessing, see [MISATO Affinity README](https://github.com/kierandidi/misato-affinity/blob/main/README.md).

## Processing Pipeline

Run scripts in the following order:

### 1. `h5_to_traj.py`
Converts HDF5 trajectory data to NetCDF (.nc) and topology (.top) files for pytraj processing.

### 2. `add_atom_residue_number.py`
Adds residue number information to each atom from trajectory topology files.

### 3. `add_binding_site_labels.py`
Labels residues as binding sites based on distance threshold (10 Å) to ligand atoms.

### 4. `compute_aligned_correlation_matrices.py`
Aligns MD trajectories and computes atom-atom correlation matrices for dynamics analysis.

### 5. `create_atom_graph_with_adaptability_for_protein_part_by_setting_thresholds_for_aligned_trajectories.py`
Builds atom-level graphs for protein regions using distance and correlation thresholds.

### 6. `create_atom_graph_BindingSiteAndLigand_with_affinity_by_setting_thresholds_for_aligned_trajectories.py`
Creates graphs containing binding site atoms and ligand atoms with binding affinity labels.

### 7. `create_residue_graph_h5_by_setting_thresholds_for_aligned_trajectories.py`
Generates residue-level graphs using Cα atoms with customizable distance and correlation thresholds.

## Directory Structure

Ensure your data directory follows this structure:

```
data/
├── downloaded/               # Direct downloads from MISATO
│   ├── adaptability_MD.hdf5
│   └── preprocessed_graph_invariant_combined.h5
├── processed/                # Processed HDF5 files
├── trajectories/             # MD trajectory files (.nc, .top)
├── correlations/
│   └── aligned_atomiccorr_no_hydrogen/
├── distances/
│   └── ca-ligand_distance/
├── maps/                     # Atom/residue type mappings
└── affinity_data.csv         # Binding affinity data
```

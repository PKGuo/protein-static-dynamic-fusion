import pytraj as pt
import numpy as np
import os
import h5py
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import shutil

def align_and_process_pdb(pdb_id, data_dir, h5_file, corr_dir, traj_dir, alignment_type='all'):
    """Process single PDB: align trajectory and compute correlation"""
    pdb_id_lower = pdb_id.lower()
    nc_file = os.path.join(data_dir, f"{pdb_id_lower}.nc")
    top_file = os.path.join(data_dir, f"{pdb_id_lower}.top")

    if not os.path.exists(nc_file) or not os.path.exists(top_file):
        return None, f"{pdb_id}: Missing NC or Top file."

    # Load and align trajectory
    traj = pt.load(nc_file, top_file)
    mask = '!@H*' if alignment_type == 'noH' else ''
    pt.align(traj, mask=mask, ref=0)
    
    # Save aligned trajectory
    aligned_nc = os.path.join(traj_dir, f"{pdb_id_lower}_aligned.nc")
    aligned_top = os.path.join(traj_dir, f"{pdb_id_lower}_aligned.top")
    pt.write_traj(aligned_nc, traj, overwrite=True)
    shutil.copy2(top_file, aligned_top)

    # Filter for correlation calculation
    traj_filtered = traj['!@H*']

    with h5py.File(h5_file, 'r') as h5f:
        if pdb_id not in h5f:
            return None, f"{pdb_id}: HDF5 dataset not found."

        pitem = h5f[pdb_id]
        cutoff = int(pitem["molecules_begin_atom_index"][:][-1])

        atom_counts = [mol.n_atoms for mol in traj_filtered.topology.mols]
        expected_counts = [int(pitem["molecules_begin_atom_index"][:][i+1]) - int(pitem["molecules_begin_atom_index"][:][i]) 
                         for i in range(len(pitem["molecules_begin_atom_index"][:]) - 1)]
        expected_counts.append(len(pitem["atoms_coordinates_ref"][:]) - cutoff)

        if atom_counts != expected_counts:
            return None, f"{pdb_id}: Atom counts do not match expected values."

        protein_atom_count = sum(atom_counts[:-1])
        ligand_atom_count = atom_counts[-1]
        total_atom_count = traj_filtered.top.n_atoms

        assert protein_atom_count == cutoff, f"Protein atom count ({protein_atom_count}) does not match cutoff ({cutoff})"

        corr_all = pt.atomiccorr(traj_filtered, byres=False)
        assert corr_all.shape == (traj_filtered.top.n_atoms, traj_filtered.top.n_atoms)

        np.savez(os.path.join(corr_dir, f"{pdb_id}_correlation_data.npz"),
                 corr_all=corr_all,
                 molecules_begin_atom_index=pitem["molecules_begin_atom_index"][:],
                 atom_count=atom_counts,
                 protein_atom_count=protein_atom_count,
                 ligand_atom_count=ligand_atom_count,
                 total_atom_count=total_atom_count)

        metadata = {
            "pdb_id": pdb_id,
            "molecules_begin_atom_index": pitem["molecules_begin_atom_index"][:].tolist(),
            "atom_count": atom_counts,
            "protein_atom_count": protein_atom_count,
            "ligand_atom_count": ligand_atom_count,
            "total_atom_count": total_atom_count
        }

        return metadata, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ids_file', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--h5_file', type=str, required=True)
    parser.add_argument('--corr_dir', type=str, required=True)
    parser.add_argument('--traj_dir', type=str, required=True)
    parser.add_argument('--alignment_type', choices=['all', 'noH'], default='all')
    args = parser.parse_args()

    os.makedirs(args.corr_dir, exist_ok=True)
    os.makedirs(args.traj_dir, exist_ok=True)

    with open(args.ids_file, 'r') as f:
        pdb_ids = f.read().splitlines()

    metadata_list = []
    for pdb_id in tqdm(pdb_ids, desc="Processing PDBs"):
        metadata, error = align_and_process_pdb(
            pdb_id, args.data_dir, args.h5_file,
            args.corr_dir, args.traj_dir, args.alignment_type
        )
        if metadata:
            metadata_list.append(metadata)
        if error:
            print(error)

    metadata_dir = os.path.join(args.corr_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    csv_file = os.path.join(metadata_dir, f"{Path(args.ids_file).stem}_metadata.csv")
    pd.DataFrame(metadata_list).to_csv(csv_file, index=False)

if __name__ == "__main__":
    main()
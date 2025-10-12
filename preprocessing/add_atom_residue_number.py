import h5py
import numpy as np
import pytraj as pt
import os
from tqdm import tqdm
import shutil

# 原始 HDF5 文件路径
original_h5_file_path = 'data/downloaded/adaptability_MD.hdf5'

# 新的 HDF5 文件路径
new_h5_file_path = 'data/processed/adaptability_MD_with_residue_numbers.hdf5'

# .nc 和 .top 文件的目录（假设文件名格式为 {pdb_code}.nc 和 {pdb_code}.top）
traj_dir = 'data/trajectories/'

# # 首先，将原始 HDF5 文件复制为新的文件
# shutil.copyfile(original_h5_file_path, new_h5_file_path)

# 打开新的 HDF5 文件进行修改
with h5py.File(new_h5_file_path, 'r+') as h5_file:
    # 获取所有的 pdb_code 列表
    pdb_codes = list(h5_file.keys())
    
    # 仅处理前 5 个数据点进行测试
    # 若要处理整个数据集，请移除切片 [:5]
    for pdb_code in tqdm(pdb_codes, desc="处理 PDB 代码"):
        # print(f"正在处理 PDB 代码：{pdb_code}")
        group = h5_file[pdb_code]
        
        # 获取 HDF5 数据中的原子数量
        total_atoms = len(group['atoms_type'])
        
        # 构建 .nc 和 .top 文件的路径
        nc_file = os.path.join(traj_dir, f"{pdb_code.lower()}.nc")
        top_file = os.path.join(traj_dir, f"{pdb_code.lower()}.top")
        
        # 检查文件是否存在
        if not os.path.isfile(nc_file):
            print(f"未找到文件：{nc_file}，跳过 {pdb_code}")
            continue
        if not os.path.isfile(top_file):
            print(f"未找到文件：{top_file}，跳过 {pdb_code}")
            continue
        
        # 使用 PyTraj 加载轨迹
        try:
            traj = pt.load(nc_file, top_file)
        except Exception as e:
            print(f"加载 {pdb_code} 的轨迹时出错：{e}")
            continue
        
        # 去除氢原子，确保与 HDF5 数据一致
        traj = traj['!@H*']
        
        # 获取 PyTraj 的残基编号
        pytraj_residue_numbers = [atom.resid for atom in traj.topology.atoms]
        
        # 检查原子数量是否匹配
        if len(pytraj_residue_numbers) != total_atoms:
            print(f"{pdb_code} 的原子数量不匹配，跳过")
            print(f"HDF5 原子数量：{total_atoms}，PyTraj 原子数量：{len(pytraj_residue_numbers)}")
            continue
        
        # 将残基编号添加到新的 HDF5 文件中
        if 'atoms_residue_number' in group:
            del group['atoms_residue_number']
        group.create_dataset('atoms_residue_number', data=pytraj_residue_numbers)
        
        # 打印结果以进行验证
        # print(f"{pdb_code} 的前 10 个残基编号：{pytraj_residue_numbers[:10]}")
        # print(f"{pdb_code} 的后 10 个残基编号：{pytraj_residue_numbers[-10:]}")

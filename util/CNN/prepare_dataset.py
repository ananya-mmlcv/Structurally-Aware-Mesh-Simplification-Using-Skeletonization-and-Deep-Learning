import os
import numpy as np
import h5py
import trimesh
import random
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import train_test_split
SEED = 448
random.seed(SEED)
np.random.seed(SEED)

# Ensure dataset is at specified path
input_path = '../../data/CNN/ModelNet40'

labels = []
# Parse dataset directory to collect label paths
for item in os.listdir(input_path):
    curr_path = os.path.join(input_path, item)
    if os.path.isdir(curr_path):
        labels.append(item)

def list_files(path, extension):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(extension)]

def load_mesh_as_voxel(path):
    mesh = trimesh.load(path, force='mesh')
    max_dimension = mesh.extents.max()
    desired_voxel_dim = 30
    pitch = max_dimension / desired_voxel_dim
    voxel_grid = mesh.voxelized(pitch=pitch)
    voxel_data = voxel_grid.matrix.astype(np.int8)
    voxels = np.zeros((32, 32, 32), dtype=np.int8)
    start_idx = [(32 - d) // 2 for d in voxel_data.shape]
    end_idx = [start + d for start, d in zip(start_idx, voxel_data.shape)]
    voxels[start_idx[0]:end_idx[0], start_idx[1]:end_idx[1], start_idx[2]:end_idx[2]] = voxel_data
    print(f'processing: {path}')
    return voxels

def prepare_dataset_paths():
    train_addrs, test_addrs, train_labels, test_labels = [], [], [], []
    for idx, item in enumerate(labels):
        train_path = os.path.join(input_path, item, 'train')
        test_path = os.path.join(input_path, item, 'test')
        train_files = list_files(train_path, '.off')
        test_files = list_files(test_path, '.off')
        train_addrs.extend(train_files)
        test_addrs.extend(test_files)
        train_labels.extend([idx] * len(train_files))
        test_labels.extend([idx] * len(test_files))
    return train_addrs, test_addrs, train_labels, test_labels

def process_mesh_data(file_path):
    return load_mesh_as_voxel(file_path)

def main():
    train_addrs, test_addrs, train_labels, test_labels = prepare_dataset_paths()
    test_addrs, val_addrs, test_labels, val_labels = train_test_split(test_addrs, test_labels, test_size=0.5)

    with Pool(cpu_count()) as pool:
        train_voxels = pool.map(process_mesh_data, train_addrs)
        test_voxels = pool.map(process_mesh_data, test_addrs)
        val_voxels = pool.map(process_mesh_data, val_addrs)

    # Setup HDF5 file for output
    with h5py.File("modelnet40_dataset.hdf5", "w") as hdf5_file:
        hdf5_file.create_dataset("train_mat", data=np.array(train_voxels, dtype=np.int8))
        hdf5_file.create_dataset("test_mat", data=np.array(test_voxels, dtype=np.int8))
        hdf5_file.create_dataset("val_mat", data=np.array(val_voxels, dtype=np.int8))
        hdf5_file.create_dataset("train_label", data=np.array(train_labels, dtype=np.int8))
        hdf5_file.create_dataset("test_label", data=np.array(test_labels, dtype=np.int8))
        hdf5_file.create_dataset("val_label", data=np.array(val_labels, dtype=np.int8))

if __name__ == "__main__":
    main()

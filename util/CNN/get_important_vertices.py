from keras.models import load_model
import h5py
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras import backend as K
import keras
import trimesh
from scipy.spatial import cKDTree
from keras.models import load_model

K.clear_session()

# Load model
autoencoder = load_model('autoencoder.h5')

def calculate_reconstruction_error(data):
    reconstructed_data = autoencoder.predict(data)
    errors = np.mean((data - reconstructed_data)**2, axis=(1, 2, 3, 4))  # Mean squared error per sample
    voxel_errors = np.square(data - reconstructed_data)  # Squared error per voxel
    return errors, voxel_errors

def voxels_from_mesh(mesh):
    max_dimension = mesh.extents.max()
    desired_voxel_dim = 30
    pitch = max_dimension / desired_voxel_dim
    voxel_grid = mesh.voxelized(pitch=pitch)
    voxel_data = voxel_grid.matrix.astype(np.int8)
    voxels = np.zeros((32, 32, 32), dtype=np.int8)
    start_idx = [(32 - d) // 2 for d in voxel_data.shape]
    end_idx = [start + d for start, d in zip(start_idx, voxel_data.shape)]
    voxels[start_idx[0]:end_idx[0], start_idx[1]:end_idx[1], start_idx[2]:end_idx[2]] = voxel_data
    return voxels

def create_voxel_centers(voxel_grid_shape, mesh_bounds):
    # Generate voxel centers within the mesh bounds
    voxel_grid = np.indices(voxel_grid_shape).reshape(3, -1).T
    voxel_size = (mesh_bounds[1] - mesh_bounds[0]) / np.array(voxel_grid_shape)
    voxel_centers = voxel_grid * voxel_size + mesh_bounds[0] + 0.5 * voxel_size
    return voxel_centers

def map_voxels_to_vertices(voxel_centers, vertices):
    # Build a KD-tree for vertices
    tree = cKDTree(vertices)
    # Query closest vertex for each voxel center
    distances, indices = tree.query(voxel_centers)
    return indices

def compute_vertex_importance(indices, voxel_importance, vertices_len):
    vertex_importance = np.zeros(vertices_len)
    for idx, voxel_idx in enumerate(indices):
        # Propagate voxel importance to vertices
        vertex_importance[voxel_idx] += voxel_importance.flat[idx]
    return vertex_importance

def save_arrays_to_txt(vertices, filename):
  with open(filename, 'w') as f:
    for vertex in vertices:
      # Convert the array to a string representation without brackets
      f.write(str(vertex) + '\n')

# Load mesh
obj_name = 'bunny'
mesh = trimesh.load(f'{obj_name}.obj', force='mesh')

mesh_voxels = voxels_from_mesh(mesh)

errors, voxel_errors = calculate_reconstruction_error(mesh_voxels.reshape([-1, box_size, box_size, box_size, 1]))

voxel_importance = voxel_errors[0].squeeze()

# Create voxel centers based on mesh bounds
voxel_centers = create_voxel_centers(voxel_importance.shape, mesh.bounds)

# Map voxel centers to closest mesh vertices
vertex_indices = map_voxels_to_vertices(voxel_centers, mesh.vertices)

# Compute importance scores for each vertex
vertex_importance = compute_vertex_importance(vertex_indices, voxel_importance, len(mesh.vertices))

# Calculate the 90th percentile value
importance_percentile = np.percentile(vertex_importance, 95)

# Find the indices of elements that are greater than the 90th percentile
important_vertices = np.where(vertex_importance > importance_percentile)[0]

print(f"Number of important vertices: {len(important_vertices)} / {len(vertex_importance)}")

save_arrays_to_txt(important_vertices, f'{obj_name}.txt')

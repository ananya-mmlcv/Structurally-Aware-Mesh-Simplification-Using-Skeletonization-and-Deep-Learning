# Voxelization and Mesh Processing

This repository contains code for voxelizing 3D meshes and processing them for use in deep learning models. The code includes loading meshes, converting them to voxel grids, preparing datasets, training a convolutional autoencoder, and visualizing the results.

## Requirements

- Python 3.x
- `numpy`
- `h5py`
- `trimesh`
- `scikit-learn`
- `keras`
- `matplotlib`
- `scipy`

You can install the required packages using pip:

```
pip install numpy h5py trimesh scikit-learn keras matplotlib scipy
```

## Overview

### Part 1: Preparing the Dataset

1. **Set Dataset Path**: Ensure the dataset is located at the specified path.
2. **Collect Labels**: Parse the dataset directory to collect label paths.
3. **List Files**: Function to list files with a specific extension.
4. **Load Mesh as Voxel**: Load a mesh file and convert it to a voxel grid.
5. **Prepare Dataset Paths**: Prepare paths for training, testing, and validation datasets.
6. **Process Mesh Data**: Function to process mesh data into voxel format.
7. **Main Function**: Load and process dataset, then save to HDF5 file.

### Part 2: Training the Autoencoder

1. **Load Data**: Load training, validation, and test data from the HDF5 file.
2. **Define Model**: Build a convolutional autoencoder model.
3. **Train Model**: Train the autoencoder with the prepared dataset.
4. **Save Model**: Save the trained model to a file.

### Part 3: Visualizing Results

1. **Load Model and Data**: Load the trained autoencoder model and test data.
2. **Predict and Apply Threshold**: Use the model to predict and apply thresholding to the results.
3. **Visualize Objects**: Function to plot and compare original and reconstructed voxel grids.
4. **Save Important Vertices**: Identify and save important vertices based on reconstruction error.

## How to Use

### Preparing the Dataset

1. **Set Folder Paths**: Update the `input_path` and `output_folder` with the paths to your dataset and desired output location.

2. **Run the Script**: Execute the script to process the mesh files and save the voxelized data to an HDF5 file.

```
python prepare_dataset.py
```

### Training the Autoencoder

1. **Ensure Dataset File**: Ensure the HDF5 file created in the previous step is available.

2. **Run the Script**: Execute the script to train the autoencoder model and save it.

```
python train_autoencoder.py
```

### Visualizing Results

1. **Ensure Model and Dataset Files**: Ensure the trained model and HDF5 dataset files are available.

2. **Run the Script**: Execute the script to visualize the original and reconstructed voxel grids, and save important vertices.

```
python visualize_results.py
```
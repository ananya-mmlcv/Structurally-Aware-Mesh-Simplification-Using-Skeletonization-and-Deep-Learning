from keras.models import load_model
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from keras import backend as K
K.clear_session()

with h5py.File('modelnet40_dataset.hdf5', 'r') as f:
    train_data = f['train_mat'][...]
    val_data = f['val_mat'][...]
    test_data = f['test_mat'][...]

train_num = train_data.shape[0]
val_num = val_data.shape[0]
test_num = test_data.shape[0]
box_size = train_data.shape[1]

train_data = train_data.reshape([-1, box_size, box_size, box_size, 1])
val_data = val_data.reshape([-1, box_size, box_size, box_size, 1])
test_data = test_data.reshape([-1, box_size, box_size, box_size, 1])

autoencoder = load_model('autoencoder.h5')
decoded_imgs = autoencoder.predict(test_data, batch_size=100)

# Apply thresholding to clean up the output
def apply_threshold(predictions, threshold=0.75):
    predictions[predictions >= threshold] = 1
    predictions[predictions < threshold] = 0
    return predictions

# Using the thresholding function on the predictions
decoded_imgs = autoencoder.predict(test_data)
decoded_imgs = apply_threshold(decoded_imgs)

decoded_imgs = decoded_imgs.reshape(test_num, box_size, box_size, box_size)
print("Decoded objects shape is:")
print(decoded_imgs.shape)

# write back to hdf5 file
hdf5_file = h5py.File("reconstruction.hdf5", "w")
hdf5_file.create_dataset("recon_mat", decoded_imgs.shape, np.int8)
for i in range(len(decoded_imgs)):
    hdf5_file["recon_mat"][i] = decoded_imgs[i]

hdf5_file.close()
print('Reconstruction HDF5 file successfully created.')

# Function to plot a single voxel grid
def plot_voxel(voxels, ax, title, important=None):
    ax.set_title(title)
    ax.voxels(voxels, facecolors='blue', edgecolor='k')
    if important is not None:
        ax.voxels(voxels * important, facecolors='red', edgecolor='k')  # Important voxels

# Load the model and data
def load_data():
    with h5py.File('/content/drive/MyDrive/object40.hdf5', 'r') as f:
        # print('keys', f.keys())
        original_data = f['test_mat'][...]
        labels = f['test_label'][...]

    with h5py.File('reconstruction.hdf5', 'r') as f:
        reconstructed_data = f['recon_mat'][...]

    return original_data, reconstructed_data, labels

label_dict = {0: "lamp", 1: "wardrobe", 2: "bench", 3: "piano", 4: "car", 5: "bed", 6: "monitor", 7: "cup", 8: "radio", 9: "mantel", 10: "desk", 11: "plant", 12: "xbox", 13: "curtain", 14: "person", 15: "stool", 16: "range_hood", 17: "guitar", 18: "chair", 19: "sink", 20: "bookshelf", 21: "dresser", 22: "cone", 23: "vase", 24: "glass_box", 25: "tent", 26: "toilet", 27: "sofa", 28: "bowl", 29: "table", 30: "flower_pot", 31: "airplane", 32: "night_stand", 33: "bathtub", 34: "tv_stand", 35: "laptop", 36: "stairs", 37: "bottle", 38: "keyboard", 39: "door"}

def visualize_objects(original_data, reconstructed_data, labels, important_voxels=None, num_samples=5):
    # Randomly select a few samples to display
    indices = np.random.choice(original_data.shape[0], num_samples, replace=False)

    fig = plt.figure(figsize=(15, num_samples * 6))

    for i, idx in enumerate(indices):
        original = original_data[idx, :, :, :].astype(bool)
        reconstructed = reconstructed_data[idx, :, :, :].astype(bool)

        # Plotting original data
        ax = fig.add_subplot(num_samples, 2, 2 * i + 1, projection='3d')

        if important_voxels is not None:
          important = important_voxels[idx, :, :, :].astype(bool)


        plot_voxel(original, ax, f'Original {label_dict[int(labels[idx])]}', important=important if important_voxels is not None else None)

        # Plotting reconstructed data
        ax = fig.add_subplot(num_samples, 2, 2 * i + 2, projection='3d')
        plot_voxel(reconstructed, ax, f'Reconstructed {label_dict[int(labels[idx])]}')

    plt.tight_layout()
    plt.savefig('comparison_1.png')  # Save the figure
    plt.show()

original_data, reconstructed_data, labels = load_data()

visualize_objects(original_data, reconstructed_data, labels)


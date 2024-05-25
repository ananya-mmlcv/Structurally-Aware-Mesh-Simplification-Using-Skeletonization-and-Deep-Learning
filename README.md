# Structurally-Aware Mesh Simplification Using Skeletonization and Deep Learning (COMP4610-2024) - README

## Overview

This project explores two novel methodologies to enhance the traditional Quadratic Error Metric (QEM) approach for mesh simplification. The integration of skeletonization and deep learning techniques aims to preserve both geometric fidelity and intrinsic structural features of 3D meshes.

We would like to credit the original Mesh simplification author that can be found [here](https://github.com/astaka-pe)

## Project Structure

```md
├── data
│   ├── output
│   └── Skeletonization
│       └── output
├── docs
└── util
```

### Directories

- **data/output/**: Contains the output files generated after mesh simplification.
- **data/Skeletonization/output/**: Contains the output files generated from the skeletonization process.
- **docs/**: Documentation files related to the project.
- **util/**: Utility scripts and modules used in the project.

### Usage

To run the mesh simplification code you need to run simplification.py

### Arguments

- `-i, --input`: Input file name (required)
- `-v`: Target vertex number
- `-p`: Rate of simplification (ignored if `-v` is specified)
- `-optim`: Specify for valence aware simplification
- `-isotropic`: Specify for isotropic simplification
- `-ix, --important_indices`: File with important vertex indices

### Example

(With Skeletonization)

```bash
python simplification.py -i data/mesh.obj -v 100 -ix data/Skeletonization/output/important_indices.txt
```

(Without Skeletonization)

```bash
python simplification.py -i data/mesh.obj -v 100
```

This command runs the mesh simplification process on `data/mesh.obj`, targeting 100 vertices, and taking into account important vertex indices from the specified file.

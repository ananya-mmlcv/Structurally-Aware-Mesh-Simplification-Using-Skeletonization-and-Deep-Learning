# Skeletonization and Mesh Simplification

This repository contains code for skeletonizing 3D meshes and converting skeleton edges to vertex pairs. The skeletonization process involves creating a simplified representation of a mesh's structure and matching these simplified vertices to the original mesh vertices.

## Requirements

- Python 3
- `skeletor`
- `networkx`
- `numpy`
- `trimesh`

You can install the required packages using pip:

```
pip install skeletor networkx numpy trimesh
```

## Overview

The provided code performs the following tasks:

1. **Load Mesh**: Loads a mesh from a given file path using `trimesh`.
2. **Skeletonize Mesh**: Skeletonizes the loaded mesh to extract skeleton vertices and edges using `skeletor`.
3. **Convert Edges**: Converts skeleton edges to vertex pairs.
4. **Match Skeleton to Mesh**: Matches skeleton vertices to the nearest mesh vertices.
5. **Build Graph**: Builds a graph from the mesh edges.
6. **Dijkstra's Shortest Path**: Uses Dijkstra's algorithm to find the shortest path in the graph.
7. **Convert Edges to Mesh Format**: Converts skeleton edges to mesh format using matched indices.
8. **Save to Text File**: Saves the important edges and vertices to a text file.

## How to Use

1. **Set Folder Paths**: Update the `folder_path` and `output_folder` with the paths to your mesh files and desired output location.

2. **Run the Script**: Execute the script to process the mesh files and save the important edges and vertices to text files.

```
if __name__ == "__main__":

    folder_path = "./computer_graphics_project/data/models"
    output_folder = "./data/Skeletonization/output"

    for obj_file in os.listdir(folder_path):

        mesh_file = os.path.join(folder_path, obj_file)
        output_vert_file = os.path.join(output_folder, obj_file.split('.')[0] +
                                    "_important_vertex.txt")
        output_edge_file = os.path.join(output_folder, obj_file.split('.')[0] +
                                    "_important_edge.txt")
        process_mesh_and_save_edges(mesh_file, output_vert_file, output_edge_file)
```

3. **Output Files**: The script will generate text files containing the important vertices and edges for each mesh file processed. These files will be saved in the specified output folder.

## Functions

### `load_mesh(file_path)`

Loads a mesh from a given file path.

### `skeletonize_mesh(mesh)`

Skeletonizes the given mesh to extract skeleton vertices and edges.

### `convert_edges_to_vertex_pairs(skeleton_edges, skeleton_vertices)`

Converts skeleton edges to vertex pairs.

### `find_nearest_point(point, vertices)`

Finds the nearest point in a list of vertices to a given point.

### `match_skeleton_to_mesh(skeleton_vertices, mesh_vertices)`

Matches skeleton vertices to the nearest mesh vertices.

### `build_graph(edges)`

Builds a graph from a list of edges.

### `dijkstra_shortest_path(graph, start, end)`

Finds the shortest path in a graph using Dijkstra's algorithm.

### `convert_edges_to_mesh_format(vert_edge, matched_index, graph)`

Converts skeleton edges to mesh format using matched indices.

### `save_list_to_txt(file_path, my_list)`

Saves a list to a text file.

### `process_mesh_and_save_edges(mesh_file, output_vert_file, output_edge_file)`

Processes the mesh file to extract skeleton and corresponding edges, then saves the important edges to a text file.

## Example Usage

To run the script, simply execute it with the appropriate folder paths set:

```
python skeletonization_script.py
```

This will process all mesh files in the specified folder and save the important vertices and edges to the output folder.

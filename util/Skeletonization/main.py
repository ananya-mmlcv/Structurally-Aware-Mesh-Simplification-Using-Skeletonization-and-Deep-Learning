import skeletor as sk
import networkx as nx
import numpy as np
import trimesh 
import heapq
import math 
import os

def load_mesh(file_path):
    """
    Load a mesh from a given file path.
    
    Parameters:
    file_path (str): Path to the mesh file.

    Returns:
    trimesh.Trimesh: Loaded mesh.
    """
    return trimesh.load_mesh(file_path)

def skeletonize_mesh(mesh):
    """
    Skeletonize the given mesh.

    Parameters:
    mesh (trimesh.Trimesh): The mesh to skeletonize.

    Returns:
    tuple: Skeleton vertices and edges.
    """
    fixed_mesh = sk.pre.fix_mesh(mesh, remove_disconnected=5, inplace=False)
    skel = sk.skeletonize.by_wavefront(fixed_mesh, waves=1, step_size=1)
    skeleton_vertices = np.array(skel.vertices)
    skeleton_edges = np.array(skel.edges)
    return skeleton_vertices, skeleton_edges

def convert_edges_to_vertex_pairs(skeleton_edges, skeleton_vertices):
    """
    Convert skeleton edges to vertex pairs.

    Parameters:
    skeleton_edges (np.ndarray): Skeleton edges.
    skeleton_vertices (np.ndarray): Skeleton vertices.

    Returns:
    np.ndarray: List of vertex pairs representing edges.
    """
    return np.array([[skeleton_vertices[edge[0]], skeleton_vertices[edge[1]]] for edge in skeleton_edges])

def find_nearest_point(point, vertices):
    """
    Find the nearest point in a list of vertices to a given point.

    Parameters:
    point (np.ndarray): The reference point.
    vertices (np.ndarray): List of vertices.

    Returns:
    tuple: Nearest point and its index.
    """
    distances = np.linalg.norm(vertices - point, axis=1)
    nearest_index = np.argmin(distances)
    nearest_point = vertices[nearest_index]
    return nearest_point, nearest_index

def match_skeleton_to_mesh(skeleton_vertices, mesh_vertices):
    """
    Match skeleton vertices to nearest mesh vertices.

    Parameters:
    skeleton_vertices (np.ndarray): Skeleton vertices.
    mesh_vertices (np.ndarray): Mesh vertices.

    Returns:
    tuple: Matched vertices and their indices.
    """
    matches = []
    matched_index = []
    for point in skeleton_vertices:
        nearest_point, nearest_index = find_nearest_point(point, mesh_vertices)
        matches.append((nearest_point, nearest_index))
        matched_index.append(nearest_index)
    return matches, matched_index

def build_graph(edges):
    """
    Build a graph from a list of edges.

    Parameters:
    edges (list): List of edges.

    Returns:
    dict: Graph represented as an adjacency list.
    """
    graph = {}
    for u, v in edges:
        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []
        graph[u].append(v)
        graph[v].append(u)
    return graph

def dijkstra_shortest_path(graph, start, end):
    """
    Find the shortest path in a graph using Dijkstra's algorithm.

    Parameters:
    graph (dict): The graph as an adjacency list.
    start (int): The starting vertex.
    end (int): The ending vertex.

    Returns:
    tuple: Length of the shortest path and the path itself.
    """
    distances = {vertex: float('inf') for vertex in graph}
    distances[start] = 0
    queue = [(0, start)]
    predecessors = {vertex: None for vertex in graph}
    while queue:
        current_distance, current_vertex = heapq.heappop(queue)
        if current_vertex == end:
            break
        if current_distance > distances[current_vertex]:
            continue
        for neighbor in graph[current_vertex]:
            distance = current_distance + 1  # Assuming all edges have unit weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor))
                predecessors[neighbor] = current_vertex
    path = []
    current_vertex = end
    while current_vertex is not None:
        path.append(current_vertex)
        current_vertex = predecessors[current_vertex]
    path.reverse()
    return distances[end], path

def convert_edges_to_mesh_format(vert_edge, matched_index, graph):
    """
    Convert skeleton edges to mesh format using matched indices.

    Parameters:
    vert_edge (list): List of vertex edges.
    matched_index (list): List of matched indices.
    graph (dict): Graph represented as an adjacency list.

    Returns:
    list: New edges in mesh format.
    """
    edge_new = []
    for v in vert_edge:
        start_vertex = v[0]
        end_vertex = v[1]
        shortest_path_length, shortest_path = dijkstra_shortest_path(graph, start_vertex, end_vertex)
        
        if math.isinf(shortest_path_length):  
            continue      
        else:
            for i in range(shortest_path_length):
                edge_new.append([shortest_path[i], shortest_path[i+1]])
    return edge_new

def save_list_to_txt(file_path, my_list):
    """
    Save a list to a text file.

    Parameters:
    file_path (str): Path to the file.
    my_list (list): List to save.
    """
    with open(file_path, 'w') as file:
        for item in my_list:
            file.write(str(item) + '\n')

def process_mesh_and_save_edges(mesh_file, output_vert_file, output_edge_file):
    """
    Process the mesh file to extract skeleton and corresponding edges, 
    then save the important edges to a text file.

    Parameters:
    mesh_file (str): Path to the mesh file.
    output_file (str): Path to the output file where important edges will be saved.
    """
    mesh = load_mesh(mesh_file)
    mesh_vertices = np.array(mesh.vertices)

    skeleton_vertices, skeleton_edges = skeletonize_mesh(mesh)
    skeleton_edges_list = convert_edges_to_vertex_pairs(skeleton_edges, skeleton_vertices)
    
    matches, matched_index = match_skeleton_to_mesh(skeleton_vertices, mesh_vertices)
    vert_edge = [[matched_index[e[0]], matched_index[e[1]]] for e in skeleton_edges]
    
    mesh_edges = np.array(mesh.edges)
    graph = build_graph(mesh_edges)
    edge_new = convert_edges_to_mesh_format(vert_edge, matched_index, graph)
    
    edges_list_new = np.array([[mesh_vertices[edge[0]], mesh_vertices[edge[1]]] for edge in edge_new])

    vertex_list = []
    for edge_ in edge_new:
        vertex_list.append(edge_[0])
        vertex_list.append(edge_[1])
    
    save_list_to_txt(output_edge_file, edge_new)
    save_list_to_txt(output_vert_file, vertex_list)

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

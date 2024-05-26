# Import necessary libraries
import numpy as np
import scipy as sp
import heapq
import copy
from tqdm import tqdm
from sklearn.preprocessing import normalize
import math

class Mesh:
    def __init__(self, path, important_indices=[], important_vertex_error=10000, build_code=False, build_mat=False, manifold=True):
        """
        Initialize a Mesh object.

        :param path: Path to the mesh file.
        :param important_indices: Indices of vertices that should be preserved during simplification.
        :param build_code: Flag to build additional structures (unused here).
        :param build_mat: Flag to build additional structures (unused here).
        :param manifold: Flag to indicate if the mesh is manifold.
        """
        # Initialize the mesh attributes
        self.path = path
        self.vs, self.faces = self.fill_from_file(path)  # Load vertices and faces from file
        self.compute_face_normals()  # Compute face normals
        self.compute_face_center()  # Compute face centers
        self.device = 'cpu'  # Device placeholder
        self.simp = False  # Simplification flag
        self.important_indices = important_indices  # Indices of important vertices
        self.important_vertex_error = important_vertex_error # Error given for vertices to be ignored (10,000 by default)
        
        # If the mesh is manifold, build additional structures
        if manifold:
            self.build_gemm()  # Build GEMM structures
            self.compute_vert_normals()  # Compute vertex normals
            self.build_v2v()  # Build vertex-to-vertex adjacency
            self.build_vf()  # Build vertex-to-face adjacency
            self.build_uni_lap()  # Build uniform Laplacian matrix

    def fill_from_file(self, path):
        """
        Load vertices and faces from a file.

        :param path: Path to the mesh file.
        :return: Tuple of vertices and faces arrays.
        """
        vs, faces = [], []  # Initialize lists for vertices and faces
        f = open(path)  # Open the file
        for line in f:  # Iterate over each line in the file
            line = line.strip()  # Remove leading/trailing whitespace
            splitted_line = line.split()  # Split line into components
            if not splitted_line:  # Skip empty lines
                continue
            elif splitted_line[0] == 'v':  # Vertex line
                # Convert vertex coordinates to floats and add to vertices list
                vs.append([float(v) for v in splitted_line[1:4]])
            elif splitted_line[0] == 'f':  # Face line
                # Extract vertex indices for the face
                face_vertex_ids = [int(c.split('/')[0]) for c in splitted_line[1:]]
                # Adjust indices to be 0-based
                face_vertex_ids = [(ind - 1) if (ind >= 0) else (len(vs) + ind) for ind in face_vertex_ids]
                if len(face_vertex_ids) == 3:
                    faces.append(face_vertex_ids)  # Add triangle face
                else:
                    # Triangulate the face if it has more than 3 vertices
                    for i in range(1, len(face_vertex_ids) - 1):
                        faces.append([face_vertex_ids[0], face_vertex_ids[i], face_vertex_ids[i + 1]])
        f.close()  # Close the file
        vs = np.asarray(vs)  # Convert vertices list to numpy array
        faces = np.asarray(faces, dtype=int)  # Convert faces list to numpy array

        # Ensure all face indices are valid
        assert np.logical_and(faces >= 0, faces < len(vs)).all()
        return vs, faces  # Return vertices and faces

    def build_gemm(self):
        """
        Build GEMM (Generalized Edge-Mesh Matrix) structures.
        """
        # Initialize adjacency and edge structures
        self.ve = [[] for _ in self.vs]  # Vertex-to-edge adjacency
        self.vei = [[] for _ in self.vs]  # Vertex-to-edge index
        edge_nb = []  # Edge neighbors
        sides = []  # Sides information
        edge2key = dict()  # Map from edge to index
        edges = []  # List of edges
        edges_count = 0  # Count of edges
        nb_count = []  # Neighbor count for each edge

        # Iterate over each face in the mesh
        for face_id, face in enumerate(self.faces):
            faces_edges = []  # List to store edges of the current face
            for i in range(3):  # Iterate over vertices in the face
                cur_edge = (face[i], face[(i + 1) % 3])  # Get the edge as a tuple
                faces_edges.append(cur_edge)  # Add the edge to the list

            # Process each edge in the face
            for idx, edge in enumerate(faces_edges):
                edge = tuple(sorted(list(edge)))  # Sort the edge vertices
                faces_edges[idx] = edge  # Update the edge in the list
                if edge not in edge2key:  # If the edge is not already in the map
                    edge2key[edge] = edges_count  # Map the edge to the current edge count
                    edges.append(list(edge))  # Add the edge to the edges list
                    edge_nb.append([-1, -1, -1, -1])  # Initialize edge neighbors
                    sides.append([-1, -1, -1, -1])  # Initialize sides information
                    self.ve[edge[0]].append(edges_count)  # Update vertex-to-edge adjacency
                    self.ve[edge[1]].append(edges_count)  # Update vertex-to-edge adjacency
                    self.vei[edge[0]].append(0)  # Update vertex-to-edge index
                    self.vei[edge[1]].append(1)  # Update vertex-to-edge index
                    nb_count.append(0)  # Initialize neighbor count for the edge
                    edges_count += 1  # Increment edge count

            # Update edge neighbors and sides information
            for idx, edge in enumerate(faces_edges):
                edge_key = edge2key[edge]  # Get the index of the current edge
                edge_nb[edge_key][nb_count[edge_key]] = edge2key[faces_edges[(idx + 1) % 3]]  # Neighboring edge 1
                edge_nb[edge_key][nb_count[edge_key] + 1] = edge2key[faces_edges[(idx + 2) % 3]]  # Neighboring edge 2
                nb_count[edge_key] += 2  # Update neighbor count

            # Update sides information
            for idx, edge in enumerate(faces_edges):
                edge_key = edge2key[edge]  # Get the index of the current edge
                sides[edge_key][nb_count[edge_key] - 2] = nb_count[edge2key[faces_edges[(idx + 1) % 3]]] - 1
                sides[edge_key][nb_count[edge_key] - 1] = nb_count[edge2key[faces_edges[(idx + 2) % 3]]] - 2

        # Convert edges, edge neighbors, and sides to numpy arrays
        self.edges = np.array(edges, dtype=np.int32)
        self.gemm_edges = np.array(edge_nb, dtype=np.int64)
        self.sides = np.array(sides, dtype=np.int64)
        self.edges_count = edges_count  # Store the total edge count

    def compute_face_normals(self):
        """
        Compute the normals for each face in the mesh.
        """
        # Calculate face normals using cross product
        face_normals = np.cross(self.vs[self.faces[:, 1]] - self.vs[self.faces[:, 0]],
                                self.vs[self.faces[:, 2]] - self.vs[self.faces[:, 0]])
        norm = np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-24  # Normalize the face normals
        face_areas = 0.5 * np.sqrt((face_normals ** 2).sum(axis=1))  # Calculate face areas
        face_normals /= norm  # Normalize face normals
        self.fn, self.fa = face_normals, face_areas  # Store face normals and areas

    def compute_vert_normals(self):
        """
        Compute the normals for each vertex in the mesh.
        """
        vert_normals = np.zeros((3, len(self.vs)))  # Initialize vertex normals
        face_normals = self.fn  # Get face normals
        faces = self.faces  # Get faces

        nv = len(self.vs)  # Number of vertices
        nf = len(faces)  # Number of faces
        mat_rows = faces.reshape(-1)  # Row indices for sparse matrix
        mat_cols = np.array([[i] * 3 for i in range(nf)]).reshape(-1)  # Column indices for sparse matrix
        mat_vals = np.ones(len(mat_rows))  # Values for sparse matrix
        f2v_mat = sp.sparse.csr_matrix((mat_vals, (mat_rows, mat_cols)), shape=(nv, nf))  # Create sparse matrix
        vert_normals = sp.sparse.csr_matrix.dot(f2v_mat, face_normals)  # Multiply face normals by sparse matrix
        vert_normals = normalize(vert_normals, norm='l2', axis=1)  # Normalize vertex normals
        self.vn = vert_normals  # Store vertex normals
    
    def compute_face_center(self):
        """
        Compute the centers for each face in the mesh.
        """
        faces = self.faces  # Get faces
        vs = self.vs  # Get vertices
        self.fc = np.sum(vs[faces], 1) / 3.0  # Calculate face centers
    
    def build_uni_lap(self):
        """
        Compute the uniform Laplacian matrix.
        """
        edges = self.edges  # Get edges
        ve = self.ve  # Get vertex-to-edge adjacency

        # Build sub-mesh vertex-vertex adjacency
        sub_mesh_vv = [edges[v_e, :].reshape(-1) for v_e in ve]
        sub_mesh_vv = [set(vv.tolist()).difference(set([i])) for i, vv in enumerate(sub_mesh_vv)]

        num_verts = self.vs.shape[0]  # Number of vertices
        mat_rows = [np.array([i] * len(vv), dtype=np.int64) for i, vv in enumerate(sub_mesh_vv)]  # Row indices
        mat_rows = np.concatenate(mat_rows)  # Concatenate row indices
        mat_cols = [np.array(list(vv), dtype=np.int64) for vv in sub_mesh_vv]  # Column indices
        mat_cols = np.concatenate(mat_cols)  # Concatenate column indices
        mat_vals = np.ones_like(mat_rows, dtype=np.float32) * -1.0  # Values for Laplacian matrix
        neig_mat = sp.sparse.csr_matrix((mat_vals, (mat_rows, mat_cols)), shape=(num_verts, num_verts))  # Create sparse matrix
        sum_count = sp.sparse.csr_matrix.dot(neig_mat, np.ones((num_verts, 1), dtype=np.float32))  # Compute sum of neighbors

        # Build diagonal matrix for Laplacian
        mat_rows_ident = np.array([i for i in range(num_verts)])
        mat_cols_ident = np.array([i for i in range(num_verts)])
        mat_ident = np.array([-s for s in sum_count[:, 0]])

        # Concatenate row, column indices and values for Laplacian matrix
        mat_rows = np.concatenate([mat_rows, mat_rows_ident], axis=0)
        mat_cols = np.concatenate([mat_cols, mat_cols_ident], axis=0)
        mat_vals = np.concatenate([mat_vals, mat_ident], axis=0)

        # Create uniform Laplacian matrix
        self.lapmat = sp.sparse.csr_matrix((mat_vals, (mat_rows, mat_cols)), shape=(num_verts, num_verts))

    def build_vf(self):
        """
        Build vertex-to-face adjacency.
        """
        vf = [set() for _ in range(len(self.vs))]  # Initialize vertex-to-face adjacency list
        for i, f in enumerate(self.faces):  # Iterate over faces
            vf[f[0]].add(i)  # Add face index to vertex 0
            vf[f[1]].add(i)  # Add face index to vertex 1
            vf[f[2]].add(i)  # Add face index to vertex 2
        self.vf = vf  # Store vertex-to-face adjacency

    def build_v2v(self):
        """
        Build vertex-to-vertex adjacency.
        """
        v2v = [[] for _ in range(len(self.vs))]  # Initialize vertex-to-vertex adjacency list
        for i, e in enumerate(self.edges):  # Iterate over edges
            v2v[e[0]].append(e[1])  # Add vertex 1 to adjacency list of vertex 0
            v2v[e[1]].append(e[0])  # Add vertex 0 to adjacency list of vertex 1
        self.v2v = v2v  # Store vertex-to-vertex adjacency

        # Compute adjacent matrix
        edges = self.edges  # Get edges
        v2v_inds = edges.T  # Transpose edges
        v2v_inds = np.concatenate([v2v_inds, v2v_inds[[1, 0]]], axis=1).astype(np.int64)  # Concatenate and transpose
        v2v_vals = np.ones(v2v_inds.shape[1], dtype=np.float32)  # Values for adjacency matrix
        self.v2v_mat = sp.sparse.csr_matrix((v2v_vals, v2v_inds), shape=(len(self.vs), len(self.vs)))  # Create sparse matrix
        self.v_dims = np.sum(self.v2v_mat.toarray(), axis=1)  # Compute vertex degrees

    def simplification(self, target_v, midpoint=False):
        """
        Simplify the mesh by reducing the number of vertices.

        :param target_v: Target number of vertices.
        :param midpoint: Flag to use midpoint for edge collapse.
        :return: Simplified mesh.
        """
        vs, vf, fn, fc, edges = self.vs, self.vf, self.fn, self.fc, self.edges  # Get mesh attributes

        # Step 1: Compute Q matrix for each vertex
        Q_s = [[] for _ in range(len(vs))]  # Initialize Q matrices
        E_s = [[] for _ in range(len(vs))]  # Initialize error values

        for i, v in enumerate(vs):  # Iterate over vertices
            f_s = np.array(list(vf[i]))  # Get faces adjacent to vertex
            fc_s = fc[f_s]  # Get face centers
            fn_s = fn[f_s]  # Get face normals
            d_s = -1.0 * np.sum(fn_s * fc_s, axis=1, keepdims=True)  # Compute d values
            abcd_s = np.concatenate([fn_s, d_s], axis=1)  # Concatenate face normals and d values
            Q_s[i] = np.matmul(abcd_s.T, abcd_s)  # Compute Q matrix for vertex
            
            # Compute error for the vertex
            v4 = np.concatenate([v, np.array([1])])
            E_s[i] = np.matmul(v4, np.matmul(Q_s[i], v4.T))
            
        # Step 2: Compute E for every possible pair of vertices and create heap
        E_heap = []
        for i, e in enumerate(edges):  # Iterate over edges

            v_0, v_1 = vs[e[0]], vs[e[1]]  # Get vertices of the edge
            Q_0, Q_1 = Q_s[e[0]], Q_s[e[1]]  # Get Q matrices of the vertices
            Q_new = Q_0 + Q_1  # Compute new Q matrix

            if midpoint:  # If using midpoint
                v_new = 0.5 * (v_0 + v_1)  # Compute midpoint
                v4_new = np.concatenate([v_new, np.array([1])])
            else:  # If not using midpoint
                Q_lp = np.eye(4)  # Initialize Q_lp matrix
                Q_lp[:3] = Q_new[:3]  # Set values of Q_lp matrix
                try:
                    Q_lp_inv = np.linalg.inv(Q_lp)  # Invert Q_lp matrix
                    v4_new = np.matmul(Q_lp_inv, np.array([[0, 0, 0, 1]]).reshape(-1, 1)).reshape(-1)
                except:  # If inversion fails, use midpoint
                    v_new = 0.5 * (v_0 + v_1)
                    v4_new = np.concatenate([v_new, np.array([1])])
            
            # Compute new error value
            E_new = np.matmul(v4_new, np.matmul(Q_new, v4_new.T))
            if (e[0] in self.important_indices) or (e[1] in self.important_indices):  # Penalize important vertices
                E_new = self.important_vertex_error # Penalize by the given error
            else:
                heapq.heappush(E_heap, (E_new, (e[0], e[1])))  # Add to heap
        
        # Step 3: Collapse minimum-error vertex
        print('done')
        simp_mesh = copy.deepcopy(self)  # Create a copy of the mesh

        vi_mask = np.ones([len(simp_mesh.vs)]).astype(np.bool_)  # Vertex inclusion mask
        fi_mask = np.ones([len(simp_mesh.faces)]).astype(np.bool_)  # Face inclusion mask

        vert_map = [{i} for i in range(len(simp_mesh.vs))]  # Vertex mapping
        pbar = tqdm(total=np.sum(vi_mask) - target_v, desc="Processing")  # Progress bar
        while np.sum(vi_mask) > target_v:  # While the number of vertices is greater than the target
            if len(E_heap) == 0:  # If the heap is empty
                print("[Warning]: edge cannot be collapsed anymore!")
                break

            E_0, (vi_0, vi_1) = heapq.heappop(E_heap)  # Pop the edge with the minimum error

            if (vi_mask[vi_0] == False) or (vi_mask[vi_1] == False):  # If either vertex is not included
                continue

            # Collapse the edge
            shared_vv = list(set(simp_mesh.v2v[vi_0]).intersection(set(simp_mesh.v2v[vi_1])))  # Shared vertices
            merged_faces = simp_mesh.vf[vi_0].intersection(simp_mesh.vf[vi_1])  # Merged faces

            if len(shared_vv) != 2:  # If non-manifold
                print(f"Skipping edge collapse due to non-manifold mesh: {len(shared_vv)} shared vertices (expected 2).")
                continue
            elif len(merged_faces) != 2:  # If boundary edge
                print("boundary edge cannot be collapsed!")
                continue
            else:
                self.edge_collapse(simp_mesh, vi_0, vi_1, merged_faces, vi_mask, fi_mask, vert_map, Q_s, E_heap)
                pbar.update(1)  # Update progress bar
        
        self.rebuild_mesh(simp_mesh, vi_mask, fi_mask, vert_map)  # Rebuild the mesh
        simp_mesh.simp = True  # Set simplification flag
        self.build_hash(simp_mesh, vi_mask, vert_map)  # Build hash structures
        
        return simp_mesh  # Return simplified mesh
    
    def edge_collapse(self, simp_mesh, vi_0, vi_1, merged_faces, vi_mask, fi_mask, vert_map, Q_s, E_heap):
        """
        Collapse an edge and update the mesh.

        :param simp_mesh: Simplified mesh.
        :param vi_0: Vertex index 0.
        :param vi_1: Vertex index 1.
        :param merged_faces: Faces to be merged.
        :param vi_mask: Vertex inclusion mask.
        :param fi_mask: Face inclusion mask.
        :param vert_map: Vertex mapping.
        :param Q_s: Q matrices.
        :param E_heap: Heap of edge errors.
        """
        shared_vv = list(set(simp_mesh.v2v[vi_0]).intersection(set(simp_mesh.v2v[vi_1])))  # Shared vertices
        new_vi_0 = set(simp_mesh.v2v[vi_0]).union(set(simp_mesh.v2v[vi_1])).difference({vi_0, vi_1})  # New vertex index 0
        simp_mesh.vf[vi_0] = simp_mesh.vf[vi_0].union(simp_mesh.vf[vi_1]).difference(merged_faces)  # Update vertex-to-face adjacency
        simp_mesh.vf[vi_1] = set()  # Clear vertex-to-face adjacency for vertex 1
        simp_mesh.vf[shared_vv[0]] = simp_mesh.vf[shared_vv[0]].difference(merged_faces)  # Update adjacency for shared vertices
        simp_mesh.vf[shared_vv[1]] = simp_mesh.vf[shared_vv[1]].difference(merged_faces)

        simp_mesh.v2v[vi_0] = list(new_vi_0)  # Update vertex-to-vertex adjacency for vertex 0
        for v in simp_mesh.v2v[vi_1]:  # Update adjacency for vertex 1
            if v != vi_0:
                simp_mesh.v2v[v] = list(set(simp_mesh.v2v[v]).difference({vi_1}).union({vi_0}))
        simp_mesh.v2v[vi_1] = []  # Clear adjacency for vertex 1
        vi_mask[vi_1] = False  # Update vertex inclusion mask

        vert_map[vi_0] = vert_map[vi_0].union(vert_map[vi_1])  # Update vertex mapping
        vert_map[vi_0] = vert_map[vi_0].union({vi_1})
        vert_map[vi_1] = set()
        
        fi_mask[np.array(list(merged_faces)).astype(np.int32)] = False  # Update face inclusion mask

        simp_mesh.vs[vi_0] = 0.5 * (simp_mesh.vs[vi_0] + simp_mesh.vs[vi_1])  # Update vertex coordinates

        # Recompute error for adjacent vertices
        Q_0 = Q_s[vi_0]
        for vv_i in simp_mesh.v2v[vi_0]:
            v_mid = 0.5 * (simp_mesh.vs[vi_0] + simp_mesh.vs[vv_i])  # Compute midpoint
            Q_1 = Q_s[vv_i]  # Get Q matrix for adjacent vertex
            Q_new = Q_0 + Q_1  # Compute new Q matrix
            v4_mid = np.concatenate([v_mid, np.array([1])])  # Concatenate midpoint
            E_new = np.matmul(v4_mid, np.matmul(Q_new, v4_mid.T))  # Compute new error
            heapq.heappush(E_heap, (E_new, (vi_0, vv_i)))  # Add to heap   
    
    @staticmethod
    def rebuild_mesh(simp_mesh, vi_mask, fi_mask, vert_map):
        """
        Rebuild the mesh after simplification.

        :param simp_mesh: Simplified mesh.
        :param vi_mask: Vertex inclusion mask.
        :param fi_mask: Face inclusion mask.
        :param vert_map: Vertex mapping.
        """
        face_map = dict(zip(np.arange(len(vi_mask)), np.cumsum(vi_mask) - 1))  # Map old to new face indices
        simp_mesh.vs = simp_mesh.vs[vi_mask]  # Update vertices
        
        vert_dict = {}
        for i, vm in enumerate(vert_map):  # Build vertex dictionary
            for j in vm:
                vert_dict[j] = i

        for i, f in enumerate(simp_mesh.faces):  # Update faces with new vertex indices
            for j in range(3):
                if f[j] in vert_dict:
                    simp_mesh.faces[i][j] = vert_dict[f[j]]

        simp_mesh.faces = simp_mesh.faces[fi_mask]  # Update faces
        for i, f in enumerate(simp_mesh.faces):  # Update face indices
            for j in range(3):
                simp_mesh.faces[i][j] = face_map[f[j]]
        
        # Recompute face normals, centers, and rebuild GEMM structures
        simp_mesh.compute_face_normals()
        simp_mesh.compute_face_center()
        simp_mesh.build_gemm()
        simp_mesh.compute_vert_normals()
        simp_mesh.build_v2v()
        simp_mesh.build_vf()

    @staticmethod
    def build_hash(simp_mesh, vi_mask, vert_map):
        """
        Build hash tables for pooling and unpooling.

        :param simp_mesh: Simplified mesh.
        :param vi_mask: Vertex inclusion mask.
        :param vert_map: Vertex mapping.
        """
        pool_hash = {}
        unpool_hash = {}
        for simp_i, idx in enumerate(np.where(vi_mask)[0]):
            if len(vert_map[idx]) == 0:
                print("[ERROR] parent node cannot be found!")
                return
            for org_i in vert_map[idx]:
                pool_hash[org_i] = simp_i
            unpool_hash[simp_i] = list(vert_map[idx])
        
        # Check for consistency
        vl_sum = 0
        for vl in unpool_hash.values():
            vl_sum += len(vl)

        if (len(set(pool_hash.keys())) != len(vi_mask)) or (vl_sum != len(vi_mask)):
            print("[ERROR] Original vertices cannot be covered!")
            return
        
        pool_hash = sorted(pool_hash.items(), key=lambda x: x[0])
        simp_mesh.pool_hash = pool_hash  # Store pool hash
        simp_mesh.unpool_hash = unpool_hash  # Store unpool hash
            
    def save(self, filename):
        """
        Save the mesh to a file.

        :param filename: Name of the file.
        """
        assert len(self.vs) > 0  # Ensure there are vertices
        vertices = np.array(self.vs, dtype=np.float32).flatten()  # Flatten vertex array
        indices = np.array(self.faces, dtype=np.uint32).flatten()  # Flatten face array

        with open(filename, 'w') as fp:
            # Write vertices
            for i in range(0, vertices.size, 3):
                x = vertices[i + 0]
                y = vertices[i + 1]
                z = vertices[i + 2]
                fp.write('v {0:.8f} {1:.8f} {2:.8f}\n'.format(x, y, z))

            # Write faces
            for i in range(0, len(indices), 3):
                i0 = indices[i + 0] + 1
                i1 = indices[i + 1] + 1
                i2 = indices[i + 2] + 1
                fp.write('f {0} {1} {2}\n'.format(i0, i1, i2))

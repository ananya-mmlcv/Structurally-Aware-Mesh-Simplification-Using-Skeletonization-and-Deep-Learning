import argparse
import os
from util.mesh import Mesh

def get_parser():
    parser = argparse.ArgumentParser(description="Mesh Simplification")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input file name")
    parser.add_argument("-v", type=int, help="Target vertex number")
    parser.add_argument("-p", type=float, default=0.1, help="Rate of simplification (Ignored by -v)")
    parser.add_argument("-optim", action="store_true", help="Specify for valence aware simplification")
    parser.add_argument("-isotropic", action="store_true", help="Specify for Isotropic simplification")
    parser.add_argument("-ix", "--important_indices", type=str, help="File with important vertex indices")
    args = parser.parse_args()
    return args

def read_indices_from_file(file_path):
    with open(file_path, 'r') as file:
        indices = []
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                try:
                    indices.append(int(line))
                except ValueError:
                    print(f"Warning: Skipping invalid line '{line}'")
    return indices

def main():
    args = get_parser()
    important_indices = []
    if args.important_indices:
        important_indices = read_indices_from_file(args.important_indices)
    
    mesh = Mesh(args.input, important_indices=important_indices)
    mesh_name = os.path.basename(args.input).split(".")[-2]
    
    if args.v:
        target_v = args.v
    else:
        target_v = int(len(mesh.vs) * args.p)
    
    if target_v >= mesh.vs.shape[0]:
        print("[ERROR]: Target vertex number should be smaller than {}!".format(mesh.vs.shape[0]))
        exit()
    
    if args.isotropic:
        simp_mesh = mesh.edge_based_simplification(target_v=target_v, valence_aware=args.optim)
    else:
        simp_mesh = mesh.simplification(target_v=target_v, valence_aware=args.optim)
    
    os.makedirs("data/output/", exist_ok=True)
    if args.important_indices:
        simp_mesh.save("data/output/{}_{}_with_skeletonization.obj".format(mesh_name, simp_mesh.vs.shape[0]))
    else:
        simp_mesh.save("data/output/{}_{}_qem.obj".format(mesh_name, simp_mesh.vs.shape[0]))
    print("[FIN] Simplification Completed!")

if __name__ == "__main__":
    main()

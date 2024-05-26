import argparse
import os
from util.mesh import Mesh

def get_parser():
    parser = argparse.ArgumentParser(description="Mesh Simplification")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input file name")
    parser.add_argument("-v", type=int, help="Target vertex number")
    parser.add_argument("-p", type=float, default=0.1, help="Rate of simplification (Ignored by -v)")
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

    simp_mesh = mesh.simplification(target_v=target_v)
    
    os.makedirs("data/output/", exist_ok=True)

        # Determine the output filename based on the input file path
    if "data/CNN" in args.important_indices:
        output_suffix = "CNN"
    else:
        output_suffix = "skeletonization" if args.important_indices else "_qem"
    
    output_filename = f"data/output/{mesh_name}_{simp_mesh.vs.shape[0]}{output_suffix}.obj"

    if args.important_indices:
        simp_mesh.save("data/output/{}_{}_with_{}.obj".format(mesh_name, simp_mesh.vs.shape[0], output_suffix))
    else:
        simp_mesh.save("data/output/{}_{}_qem.obj".format(mesh_name, simp_mesh.vs.shape[0]))
    print("[FIN] Simplification Completed!")

if __name__ == "__main__":
    main()

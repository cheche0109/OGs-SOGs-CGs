
import numpy as np
import csv
from multiprocessing import Pool


# Function to compute pairwise distances efficiently
def compute_distance(args):
    tree, leaves, i, j = args
    return i, j, tree.get_distance(leaves[i], leaves[j])

# Function to compute and save the pairwise distance matrix
def compute_and_save_distance_matrix(tree_file, output_file):
    from ete3 import Tree

    # Read tree from Newick file
    def load_tree_from_file(file_path):
        with open(file_path, 'r') as f:
            newick_tree = f.read().strip()
        return Tree(newick_tree)


    tree = load_tree_from_file(tree_file)
    leaves = tree.get_leaf_names()
    n = len(leaves)

    dist_matrix = np.zeros((n, n))

    # Use multiprocessing to speed up distance computation
    args = [(tree, leaves, i, j) for i in range(n) for j in range(i+1, n)]
    with Pool(processes=24) as pool:
        results = pool.map(compute_distance, args)

    for i, j, dist in results:
        dist_matrix[i, j] = dist
        dist_matrix[j, i] = dist  # Symmetric matrix

    # Save distance matrix to file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Leaf"] + leaves)
        for i, row in enumerate(dist_matrix):
            writer.writerow([leaves[i]] + list(row))
    print(f"Distance matrix saved to {output_file}")

    #for i, leaf1 in enumerate(leaves):
    #    print(i)
    #    for j, leaf2 in enumerate(leaves):
    #        if i < j:
    #            dist = tree.get_distance(leaf1, leaf2)
    #            dist_matrix[i, j] = dist
    #            dist_matrix[j, i] = dist  # Symmetric matrix

    # Save distance matrix to file
    #with open(output_file, 'w', newline='') as f:
    #    writer = csv.writer(f)
    #    writer.writerow(["Leaf"] + leaves)
    #    for i, row in enumerate(dist_matrix):
    #        writer.writerow([leaves[i]] + list(row))
    #print(f"Distance matrix saved to {output_file}")

def load_distance_matrix(file_path):
    """Load distance matrix from a CSV file."""
    with open(file_path, 'r') as f:
        reader = list(csv.reader(f))

    leaves = reader[0][1:]
    dist_matrix = np.array([[float(value) for value in row[1:]] for row in reader[1:]])
    return leaves, dist_matrix

def perform_mds(distance_matrix, leaves, n_components, output_csv):
    """Perform MDS and save the coordinates."""

    from sklearn.manifold import MDS
    #from cuml.manifold import MDS

    mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42) #takes very long
    #mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42, n_init=1, max_iter=100)
    #mds = MDS(n_components=n_components, dissimilarity='precomputed') #cuml
    coords = mds.fit_transform(distance_matrix)
    stress = mds.stress_
    print(f"MDS Stress: {stress}")

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ["Leaf", "MDS1", "MDS2"] if n_components == 2 else ["Leaf", "MDS1", "MDS2", "MDS3"]
        writer.writerow(header)
        for i, name in enumerate(leaves):
            writer.writerow([name] + list(coords[i]))
    print(f"MDS coordinates saved to {output_csv}")

    return coords

def plot_mds(coords, leaves, n_components=2):
    """Plot the MDS projection."""

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # Ensure 3D plotting is supported

    fig = plt.figure(figsize=(8, 6))
    if n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], color='b')
        for i, name in enumerate(leaves):
            ax.text(coords[i, 0], coords[i, 1], coords[i, 2], name)
        ax.set_zlabel("MDS3")
    else:
        ax = fig.add_subplot(111)
        ax.scatter(coords[:, 0], coords[:, 1], color='b')
        for i, name in enumerate(leaves):
            ax.text(coords[i, 0], coords[i, 1], name)

    ax.set_xlabel("MDS1")
    ax.set_ylabel("MDS2")
    plt.title("Projection of Phylogenetic Distances")

    output_file = "phylogeny_projection.png"
    plt.savefig(output_file, dpi=300)
    print(f"Figure saved as {output_file}")



# Replace 'tree.nwk' with your Newick file path
#tree_file = 'phylogenies/ar122_iqtree.nwk'
tree_file = 'phylogenies/bac120_iqtree.nwk'
nd=3
#compute_and_save_distance_matrix(tree_file,"{}_dist".format(tree_file))
leaves, dist_matrix = load_distance_matrix("{}_dist".format(tree_file))
perform_mds(dist_matrix,leaves,nd,"{}_coordinates_{}d.csv".format(tree_file,nd))

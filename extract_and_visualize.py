import os
import open3d as o3d
import numpy as np
import torch
import umap
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load a .ply file and return point cloud data
def load_ply(file_path):
    point_cloud = o3d.io.read_point_cloud(file_path)
    points = np.asarray(point_cloud.points)
    return points

# Function to extract features using PointNet
def extract_features(point_cloud, model):
    #### TODO: USE MODEL 
    pass

# Function to visualize UMAP projection and save as image
def visualize_umap(features, output_path):
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=2, random_state=42)
    embedding = reducer.fit_transform(features)

    plt.figure(figsize=(12, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], s=5, cmap='Spectral')
    plt.title("UMAP Projection of Extracted Features")
    plt.savefig(output_path, format='jpg')
    print(f"UMAP projection saved to {output_path}")

def main():
    #### TODO: LOAD MODEL

    # Directory containing the .ply files
    ply_dir = "output/"
    ply_files = [f for f in os.listdir(ply_dir) if f.endswith('.ply')]

    features_list = []
    
    # Loop through all ply files and extract features
    for ply_file in ply_files:
        file_path = os.path.join(ply_dir, ply_file)
        print(f"Processing {file_path}")
        point_cloud = load_ply(file_path)
        features = extract_features(point_cloud, model)
        features_list.append(features)

    # Convert the features list to a NumPy array
    features_array = np.vstack(features_list)

    # Perform UMAP projection and save the result
    output_umap_path = os.path.join(ply_dir, "umap_projection.jpg")
    visualize_umap(features_array, output_umap_path)

if __name__ == "__main__":
    main()

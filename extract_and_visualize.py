import os
import sys
import numpy as np
import torch
import open3d as o3d
import MinkowskiEngine as ME
import matplotlib.pyplot as plt
from umap import UMAP
from omegaconf import DictConfig
import hydra

home_dir = os.path.abspath(os.path.dirname(__file__))
# Add PointContrast directory to system path for imports
pointcontrast_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'third_party/PointContrast'))
sys.path.append(os.path.join(pointcontrast_dir, 'pretrain', 'pointcontrast'))

# Import the pre-trained model and configuration
from third_party.PointContrast.pretrain.pointcontrast.model.res16unet import Res16UNet34C
from third_party.PointContrast.downstream.votenet_det_new.models.backbone.sparseconv.config import get_config

def visualize_umap(features, ply_files, output_path):
    """
    Perform UMAP dimensionality reduction and create an interactive visualization.
    Clicking on a point in the plot will open the corresponding PLY file in a window.

    Args:
        features (numpy.ndarray): Array of shape (num_instances, feature_dim).
        ply_files (list): List of PLY file paths corresponding to each instance.
        output_path (str): Path to save the UMAP visualization image.
    """
    # Initialize UMAP reducer
    reducer = UMAP(n_neighbors=5, min_dist=0.3, n_components=2, random_state=42)
    # Fit and transform the features
    embedding = reducer.fit_transform(features)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create a scatter plot of the embeddings with pickable points
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], s=50, cmap='Spectral', picker=True)
    ax.set_title("UMAP Projection of Averaged Instance Features")

    # Define the pick event handler
    def onpick(event):
        ind = event.ind[0]  # Index of the picked point
        ply_file = ply_files[ind]
        print(f"Clicked on point {ind}, corresponding to file {ply_file}")

        # Load the PLY file
        pcd = o3d.io.read_point_cloud(ply_file)

        # Open the point cloud using Open3D's visualization window
        o3d.visualization.draw_geometries([pcd], window_name=f"Instance {ind}", width=800, height=600)

    # Connect the pick event handler
    fig.canvas.mpl_connect('pick_event', onpick)

    # Save the figure
    plt.savefig(output_path, format='jpg')
    print(f"UMAP projection saved to {output_path}")

    # Show the interactive plot
    plt.show()

@hydra.main(config_path="third_party/PointContrast/pretrain/pointcontrast/config/", config_name="defaults.yaml")
def main(cfg: DictConfig) -> None:
    """
    Main function to extract features from point clouds and visualize using UMAP.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    # Load the configuration
    config = cfg

    # Ensure working directory is correct
    os.chdir(home_dir)

    # Directory containing PLY files
    ply_dir = os.path.join(home_dir, "output")
    
    # List all PLY files in the directory
    ply_files = [f for f in os.listdir(ply_dir) if f.endswith('.ply')]

    # Initialize the pre-trained model
    in_channels = 3       # Input feature dimension (e.g., XYZ coordinates or RGB colors)
    out_channels = 32     # Output feature dimension as per pretraining
    model = Res16UNet34C(in_channels=in_channels, out_channels=out_channels, config=config)

    # Load the pre-trained checkpoint
    checkpoint_path = 'data/checkpoints/nce.pth'  # Adjust the path if necessary
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']

    # Remove 'module.' prefix from state_dict keys if present
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[len('module.'):] if k.startswith('module.') else k
        new_state_dict[name] = v

    # Load state_dict into the model
    model.load_state_dict(new_state_dict)
    model.eval()

    # List to store averaged features per instance
    averaged_features_list = []

    # Full paths of the PLY files
    ply_file_paths = []

    # Loop through all PLY files and extract features
    for i, ply_file in enumerate(ply_files):
        file_path = os.path.join(ply_dir, ply_file)
        ply_file_paths.append(file_path)
        print(f"Processing {file_path}")

        # Load point cloud
        point_cloud = o3d.io.read_point_cloud(file_path)
        points = np.asarray(point_cloud.points)

        # Convert points to torch.Tensor
        coords = torch.from_numpy(points).float()

        # Use color information as features if available; otherwise, use XYZ coordinates
        if point_cloud.has_colors():
            # Normalize colors to [0, 1] if necessary
            feats = torch.from_numpy(np.asarray(point_cloud.colors, dtype=np.float32))
        else:
            feats = coords  # Use XYZ coordinates as features

        print(f"Features shape before model: {feats.shape}")  # Should be [num_points, 3]

        # Create MinkowskiEngine SparseTensor
        sparse_tensor = ME.SparseTensor(
            feats,
            ME.utils.batched_coordinates([coords])
        )

        # Extract features using the model
        with torch.no_grad():
            output = model(sparse_tensor)

        # Get the features from the output SparseTensor
        features = output.F  # Features of shape [num_points, out_channels]

        print(f"Features shape after model: {features.shape}")  # Should be [num_points, out_channels]

        # Average features over all points to get a single feature vector per instance
        averaged_feature = features.mean(dim=0)  # Shape: [out_channels]
        averaged_features_list.append(averaged_feature.numpy())

    # Convert the list of averaged features to a NumPy array
    averaged_features_array = np.vstack(averaged_features_list)  # Shape: [num_instances, out_channels]

    # Perform UMAP projection on the instance-level features and save the result
    output_umap_path = os.path.join(ply_dir, "umap_projection.jpg")
    visualize_umap(averaged_features_array, ply_file_paths, output_umap_path)

if __name__ == "__main__":
    main()

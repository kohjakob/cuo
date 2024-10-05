import os
import torch
import open3d as o3d
import numpy as np
import albumentations as A
import MinkowskiEngine as ME
import matplotlib.pyplot as plt
import umap
import json
import colorsys
import hydra
from omegaconf import DictConfig
import sys
import torch
#from torch_points3d.models.classification.pointnet import PointNet

home_dir = os.path.abspath(os.path.dirname(__file__))
unscene_dir = os.path.join(home_dir, 'third_party', 'UnScene3D')
sys.path.append(unscene_dir)

from third_party.UnScene3D.utils.utils import load_checkpoint_with_missing_or_exsessive_keys
from third_party.UnScene3D.datasets.scannet200.scannet200_constants import CLASS_LABELS_200
from third_party.UnScene3D.models.mask3d import Mask3D
#from third_party.UnScene3D.trainer.trainer import InstanceSegmentation

# Config paths
scannet_scene_dir = 'data/scannet_scenes/'
output_dir = 'output/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
unscene3d_weights_path = os.path.join(home_dir, 'data/checkpoints/best2.ckpt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
@hydra.main(config_path="third_party/UnScene3D/conf", config_name="config_base_instance_segmentation.yaml")
def main(cfg: DictConfig):

    model = Instance_Segmentation_Model(cfg)
    normalize_color = A.Normalize(
        mean=(0.47793125906962, 0.4303257521323044, 0.3749598901421883),
        std=(0.2834475483823543, 0.27566157565723015, 0.27018971370874995)
    )

    features_list = []
    ground_truth_labels_list = []
    scene_names = ['scene0000_00', 'scene0001_00', 'scene0002_00', 'scene0003_00', 'scene0004_00']

    for scene_name in scene_names:
        print(f'Processing {scene_name}')
        
        outputs, inverse_map, coords, colors_normalized, ground_truth_labels = process_scene(scene_name, model, normalize_color)
        
        logits = outputs["pred_logits"]
        masks = outputs["pred_masks"]

        # Reformat predictions
        logits = logits[0].detach().cpu()
        masks = masks[0].detach().cpu()

        confidences = []
        masks_binary = []
        labels = []
        scores = []

        for i in range(len(logits)):
            p_labels = torch.softmax(logits[i], dim=-1)
            p_masks = torch.sigmoid(masks[:, i])
            l = torch.argmax(p_labels, dim=-1)
            c_label = torch.max(p_labels)
            m = p_masks > 0.5
            c_m = p_masks[m].sum() / (m.sum() + 1e-8)
            c = c_m

            if l < 200 and c > 0.7:
                confidences.append(c.item())
                masks_binary.append(m.numpy()[inverse_map])
                labels.append(l.item())
                scores.append(c_label.item())

        masks_binary, confidences = merge_instances(masks_binary, confidences)
        masks_binary, confidences = filter_by_confidence(masks_binary, confidences, 0.5)

        if masks_binary is not None:
            
            #pretrained_model = ShapeNetPretrainedModel(checkpoint_path='path/to/shapenet_pretrained_weights.pth')
            #pretrained_model = pretrained_model.to(device)
            #features, gt_labels = extract_features_of_merged_instances2(masks_binary, coords, colors_normalized, ground_truth_labels, inverse_map, pretrained_model)
            
            features, gt_labels = extract_features_of_merged_instances(masks_binary, coords, colors_normalized, ground_truth_labels, inverse_map, model)
            
            features_list.extend(features)
            ground_truth_labels_list.extend(gt_labels)

    features_array = np.array(features_list)
    ground_truth_labels_array = np.array(ground_truth_labels_list)

    if len(features_array) > 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        visualize_umap(features_array, ground_truth_labels_array, os.path.join(output_dir, 'umap_projection.png'))
    else:
        print("No features extracted to visualize.")
    
    
    
    
    
    


class ShapeNetPretrainedModel(torch.nn.Module):
    def __init__(self, checkpoint_path=None):
        super(ShapeNetPretrainedModel, self).__init__()
        self.model = PointNet(num_classes=40)  # Assuming ShapeNet has 40 classes
        
        if checkpoint_path is not None:
            self.load_pretrained_weights(checkpoint_path)
    
    def load_pretrained_weights(self, checkpoint_path):
        print(f"Loading pretrained model from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path)
        self.model.load_state_dict(state_dict)

    def forward(self, x):
        return self.model(x)
    
# Extract features of merged instances
def extract_features_of_merged_instances2(masks_binary, coords, colors_normalized, ground_truth_labels, inverse_map, pretrained_model):
    features_list = []
    ground_truth_labels_list = []

    for mask in masks_binary:
        instance_point_indices = np.where(mask == 1)[0]
        if len(instance_point_indices) == 0:
            continue

        instance_coords = coords[instance_point_indices]
        instance_features = colors_normalized[instance_point_indices]
        instance_gt_labels = ground_truth_labels[instance_point_indices]

        # Convert to PyTorch tensors
        instance_coords_tensor = torch.from_numpy(instance_coords).int().to(device)
        instance_features_tensor = torch.from_numpy(instance_features).float().to(device)

        # Pass the instance through the ShapeNet-pretrained model
        with torch.no_grad():
            # Here you use the pretrained ShapeNet model to extract features
            instance_feats = pretrained_model(instance_features_tensor)
            instance_feats = instance_feats.mean(dim=0).cpu().numpy()

        # Assign the majority label
        unique_labels_instance, counts = np.unique(instance_gt_labels, return_counts=True)
        majority_label = unique_labels_instance[np.argmax(counts)]

        # Append features and ground truth labels
        features_list.append(instance_feats)
        ground_truth_labels_list.append(majority_label)

    return features_list, ground_truth_labels_list
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   
class Instance_Segmentation_Model(torch.nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        cfg.general.checkpoint = unscene3d_weights_path
        cfg.general.train_on_segments = False
        cfg.general.eval_on_segments = False
        #self.model = InstanceSegmentation(cfg)
        self.model = hydra.utils.instantiate(cfg.model)
        os.chdir(home_dir)
        

        
        #state_dict = torch.load('data/checkpoints/best.ckpt')["state_dict"]
        #self.model.load_state_dict(state_dict)
        _, self.model = load_checkpoint_with_missing_or_exsessive_keys(cfg, self.model)
        self.model = self.model.to(device)
        self.model.eval()  
        os.chdir(home_dir)

    def forward(self, x, raw_coordinates=None):
        return self.model(x, raw_coordinates=raw_coordinates)

# Function for processing each scene
def process_scene(scene_name, model, normalize_color):
    scene_dir = os.path.join(scannet_scene_dir, scene_name)
    mesh_file = os.path.join(scene_dir, f'{scene_name}_vh_clean.ply')

    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()

    points = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)

    # Load segmentation data
    segs_file = os.path.join(scene_dir, f'{scene_name}_vh_clean.segs.json')
    aggr_file = os.path.join(scene_dir, f'{scene_name}_vh_clean.aggregation.json')

    # Load segmentation indices and aggregation data
    g = os.getcwd()
    with open(segs_file, 'r') as f:
        segs_json = json.load(f)
    seg_indices = np.array(segs_json['segIndices'])

    with open(aggr_file, 'r') as f:
        aggr_json = json.load(f)
    seg_groups = aggr_json['segGroups']

    # Map segment IDs to label IDs
    segid_to_label = {}
    for seg_group in seg_groups:
        label = seg_group['label']
        segments = seg_group['segments']
        label_id = CLASS_LABELS_200.index(label) if label in CLASS_LABELS_200 else -1
        for segid in segments:
            segid_to_label[segid] = label_id

    # Assign ground truth labels to points
    ground_truth_labels = np.array([segid_to_label.get(segid, -1) for segid in seg_indices])

    # Filter out unlabeled points
    valid_idx = ground_truth_labels != -1
    points = points[valid_idx]
    colors = colors[valid_idx]
    ground_truth_labels = ground_truth_labels[valid_idx]

    # Normalize colors
    colors = colors * 255.
    pseudo_image = colors.astype(np.uint8)[np.newaxis, :, :]
    colors_normalized = np.squeeze(normalize_color(image=pseudo_image)["image"])

    # Voxelization
    coords = np.floor(points / 0.02)

    _, _, unique_map, inverse_map = ME.utils.sparse_quantize(
        coordinates=coords, features=colors_normalized, return_index=True, return_inverse=True
    )

    sample_coordinates = coords[unique_map]
    coordinates = [torch.from_numpy(sample_coordinates).int()]
    sample_features = colors_normalized[unique_map]
    features = [torch.from_numpy(sample_features).float()]
    sample_gt_labels = ground_truth_labels[unique_map]

    coordinates, _ = ME.utils.sparse_collate(coords=coordinates, feats=features)
    features = torch.cat(features, dim=0)
    data = ME.SparseTensor(
        coordinates=coordinates,
        features=features,
        device=device,
    )

    # Run model
    with torch.no_grad():
        outputs = model(data, raw_coordinates=features)
    del data
    torch.cuda.empty_cache()
        
    return outputs, inverse_map, coords, colors_normalized, ground_truth_labels

def merge_instances(masks_binary, confidences):
    num_instances = len(masks_binary)
    if num_instances <= 1:
        return masks_binary, confidences

    merging = True
    while merging and num_instances > 1:
        merging = False
        iou_matrix = np.zeros((num_instances, num_instances))
        for i in range(num_instances):
            for j in range(i + 1, num_instances):
                mask_i = masks_binary[i]
                mask_j = masks_binary[j]
                intersection = np.logical_and(mask_i, mask_j).sum()
                union = np.logical_or(mask_i, mask_j).sum()
                iou_matrix[i, j] = intersection / union if union > 0 else 0.0
                iou_matrix[j, i] = iou_matrix[i, j]

        pairs_to_merge = np.argwhere(iou_matrix > 0.5)
        pairs_to_merge = pairs_to_merge[pairs_to_merge[:, 0] < pairs_to_merge[:, 1]]

        if len(pairs_to_merge) > 0:
            merging = True
            parent = np.arange(num_instances)

            def find(u):
                while parent[u] != u:
                    parent[u] = parent[parent[u]]
                    u = parent[u]
                return u

            def union(u, v):
                pu, pv = find(u), find(v)
                if pu != pv:
                    parent[pu] = pv

            for u, v in pairs_to_merge:
                union(u, v)

            root_to_indices = {}
            for idx in range(num_instances):
                root = find(idx)
                if root not in root_to_indices:
                    root_to_indices[root] = []
                root_to_indices[root].append(idx)

            merged_masks = []
            merged_confidences = []

            for indices in root_to_indices.values():
                masks_binary = np.array(masks_binary)  # Convert to NumPy array
                merged_mask = np.logical_or.reduce(masks_binary[indices, :], axis=0)

                # Convert confidences to a NumPy array for proper indexing
                confidences_np = np.array(confidences)

                # Select confidences corresponding to the indices
                comp_confidences = confidences_np[indices]  # Now you can index with 'indices'
                
                # Find the index of the highest confidence and select the corresponding confidence
                max_conf_idx = indices[np.argmax(comp_confidences)]
                selected_confidence = confidences[max_conf_idx]

                merged_masks.append(merged_mask)
                merged_confidences.append(selected_confidence)

            masks_binary = np.array(merged_masks)
            confidences = np.array(merged_confidences)
            num_instances = len(masks_binary)

    return masks_binary, confidences


# Function to filter instances by confidence
def filter_by_confidence(masks_binary, confidences, threshold=0.9):
    valid_indices = np.where(np.array(confidences) > threshold)[0]
    if len(valid_indices) == 0:
        return None, None

    masks_binary = masks_binary[valid_indices]
    confidences = confidences[valid_indices]
    return masks_binary, confidences

# Function to extract features of merged instances
def extract_features_of_merged_instances(masks_binary, coords, colors_normalized, ground_truth_labels, inverse_map, model):
    features_list = []
    ground_truth_labels_list = []

    for mask in masks_binary:
        instance_point_indices = np.where(mask == 1)[0]
        if len(instance_point_indices) == 0:
            continue

        instance_coords = coords[instance_point_indices]
        instance_features = colors_normalized[instance_point_indices]
        instance_gt_labels = ground_truth_labels[instance_point_indices]

        instance_coords_tensor = torch.from_numpy(instance_coords).int().to(device)
        instance_features_tensor = torch.from_numpy(instance_features).float().to(device)

        instance_sparse_tensor = ME.SparseTensor(
            coordinates=ME.utils.batched_coordinates([instance_coords_tensor]),
            features=instance_features_tensor,
            device=device,
        )

        with torch.no_grad():
            backbone_output = model.model.backbone(instance_sparse_tensor)
            instance_feats, _ = backbone_output
            instance_feats = instance_feats.F

        feature_vector = instance_feats.mean(dim=0).cpu().numpy()
        unique_labels_instance, counts = np.unique(instance_gt_labels, return_counts=True)
        majority_label = unique_labels_instance[np.argmax(counts)]

        features_list.append(feature_vector)
        ground_truth_labels_list.append(majority_label)

    return features_list, ground_truth_labels_list

# Function to generate N distinct colors
def generate_distinct_colors(n):
    """Generate n distinct colors using the HSV color space."""
    hues = np.linspace(0, 1, n + 1)[:-1]
    colors = [colorsys.hsv_to_rgb(h, 0.65, 0.95) for h in hues]
    return colors

# UMAP visualization function
def visualize_umap(features, ground_truth_labels, output_path):
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=2, random_state=42)
    embedding = reducer.fit_transform(features)

    unique_labels = np.unique(ground_truth_labels)
    num_classes = len(unique_labels)
    colors = generate_distinct_colors(num_classes)

    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}

    plt.figure(figsize=(12, 8))
    for i, label in enumerate(unique_labels):
        idx = np.where(ground_truth_labels == label)
        plt.scatter(embedding[idx, 0], embedding[idx, 1],
                    label=CLASS_LABELS_200[label],
                    color=label_to_color[label], alpha=0.7, s=20)

    plt.legend(loc="best", fontsize='xx-small', bbox_to_anchor=(1.05, 1), ncol=2)
    plt.title('UMAP Projection of Instance Features (Ground Truth Classes)')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f'Saved UMAP visualization to {output_path}')


if __name__ == '__main__':
    main()

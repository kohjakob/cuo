## Setup
### Environment Setup using `environment.yml`

Create conda environment:
```
conda env create -f conf/environment.yml
```

Activate the environment:
```
conda activate cuo
```

### MinkowskiEngine and Detectron2 (e.g. with CUDA 11.6 in /usr/local/cuda-11.6)
```
export CUDA_HOME=/usr/local/cuda-11.6
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### Custom Cpp/Cuda tools
```
cd third_party/UnScene3D/utils/cpp_utils && python setup.py install
cd ../cuda_utils && python setup.py install
cd ../../third_party/pointnet2 && python setup.py install
cd ../../../..
```

### Download first 5 scannet scenes (Adjust parameters: first, last scene id)
```
mkdir data/scannet_scenes/
bash data/download_scannet_subset.sh 0 4
```

### Download UnScene3D weights
```
mkdir data/checkpoints/
```
Download weights from https://kaldir.vc.in.tum.de/unscene3d/checkpoints/unscene3d_DINO_CSC_self_train_3.zip
Extract
Move best.ckpt to data/checkpoints/


## Running
Segmentation, merging by IoU and saving individual instances to .ply
```
python3 segment_and_merge.py
```
TODO: Feature extraction, umap projection, feature extraction and clustering
```
python3 extract_and_visualize.py
```
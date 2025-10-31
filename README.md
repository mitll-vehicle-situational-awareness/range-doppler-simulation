# Radar Point Cloud Simulation

## Deep Learning 3D Reconstruction

This repo now includes a simple deep-learning pipeline that maps radar point clouds to a dense 3D occupancy volume using a lightweight 3D UNet.

### Install

Install dependencies
```bash
mamba env create -f environment.yml
```
Initialize mamba hook:
```bash
eval "$(mamba shell hook --shell zsh)"
```
Activate environment:
```bash
mamba activate radar-pc
```

If you have a CUDA GPU, install a CUDA-specific PyTorch build from the official site for best performance.

### Data

- Expects per-frame CSVs under `sample_data/` named like `pointcloud_frame_*.csv` with columns:
  - `X,Y,Z` (meters), optional `Intensity,Vx,Vy,Vz` ignored by the DL pipeline.

### Train

Trains a denoising/self-supervised occupancy model where targets are the voxelized input occupancy.

```bash
python -m models.dl_reconstruction train \
  --data-dir sample_data \
  --nx 128 --ny 128 --nz 32 \
  --xmin -50 --xmax 50 --ymin -50 --ymax 50 --zmin -5 --zmax 5 \
  --epochs 10 --batch-size 2 --lr 1e-3 \
  --out-dir checkpoints
```

This saves per-epoch checkpoints and `unet3d_best.pt`.

### Inference

```bash
python -m models.dl_reconstruction infer \
  --input sample_data \
  --checkpoint checkpoints/unet3d_best.pt \
  --nx 128 --ny 128 --nz 32 \
  --xmin -50 --xmax 50 --ymin -50 --ymax 50 --zmin -5 --zmax 5 \
  --thresh 0.4 \
  --out-dir dl_outputs
```

Generates PNG visualizations of occupied voxels per frame.

### Classical Visualization

To visualize the raw point cloud in 3D without DL:

```bash
python -m models.radar --input sample_data --format csv
```


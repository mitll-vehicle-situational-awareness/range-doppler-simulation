"""
Deep Learning 3D Reconstruction from Radar Point Clouds

This module provides:
- RadarDataset: loads per-frame CSV point clouds to voxel grids
- UNet3D: lightweight 3D UNet for occupancy prediction
- Train/eval/infer CLIs

Notes
- This is a scaffold you can train on your data. It maps sparse point clouds
  to dense occupancy voxels. Labels can be self-generated (binary occupancy of
  observed points) or externally provided if you have GT meshes/voxels.
"""

import os
import glob
import math
import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def points_to_voxels(points: np.ndarray,
                     bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
                     grid_size: Tuple[int, int, int]) -> np.ndarray:
    """
    Rasterize XYZ points into a binary voxel grid.

    Args:
        points: (N, 3) array in meters [x, y, z]
        bounds: ((xmin,xmax), (ymin,ymax), (zmin,zmax)) in meters
        grid_size: (nx, ny, nz)
    Returns:
        voxels: (1, nz, ny, nx) binary occupancy grid (channel-first, zyx)
    """
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    nx, ny, nz = grid_size
    if points.size == 0:
        return np.zeros((1, nz, ny, nx), dtype=np.float32)

    # Normalize to [0, 1)
    px = (points[:, 0] - xmin) / max(1e-6, (xmax - xmin))
    py = (points[:, 1] - ymin) / max(1e-6, (ymax - ymin))
    pz = (points[:, 2] - zmin) / max(1e-6, (zmax - zmin))

    # Convert to indices
    ix = np.clip((px * nx).astype(int), 0, nx - 1)
    iy = np.clip((py * ny).astype(int), 0, ny - 1)
    iz = np.clip((pz * nz).astype(int), 0, nz - 1)

    vox = np.zeros((nz, ny, nx), dtype=np.float32)
    vox[iz, iy, ix] = 1.0
    return vox[None, ...]


class RadarDataset(Dataset):
    """Loads per-frame radar CSVs as voxel grids.

    CSV expected columns: header + X,Y,Z,(Intensity,Vx,Vy,Vz optional)
    """

    def __init__(self,
                 data_dir: str,
                 pattern: str = "pointcloud_frame_*.csv",
                 bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((-50, 50), (-50, 50), (-5, 5)),
                 grid_size: Tuple[int, int, int] = (128, 128, 32),
                 augment: bool = False):
        self.files = sorted(glob.glob(os.path.join(data_dir, pattern)))
        if not self.files:
            raise ValueError(f"No CSV files found in {data_dir} matching {pattern}")
        self.bounds = bounds
        self.grid_size = grid_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.files)

    def _load_points(self, csv_path: str) -> np.ndarray:
        data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
        if data.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return data[:, :3].astype(np.float32)

    def _random_flip(self, pts: np.ndarray) -> np.ndarray:
        if pts.size == 0:
            return pts
        if np.random.rand() < 0.5:
            pts[:, 0] = -pts[:, 0]
        if np.random.rand() < 0.5:
            pts[:, 1] = -pts[:, 1]
        return pts

    def __getitem__(self, idx: int):
        pts = self._load_points(self.files[idx])
        if self.augment:
            pts = self._random_flip(pts)

        # Input: sparse occupancy; Target: same occupancy (denoising/self-supervised)
        vox = points_to_voxels(pts, self.bounds, self.grid_size)  # (1, Z, Y, X)

        # Corrupt input slightly for denoising target
        inp = vox.copy()
        if self.augment and np.random.rand() < 0.8:
            noise_mask = (np.random.rand(*inp.shape) < 0.02).astype(np.float32)
            inp = np.clip(inp + noise_mask, 0.0, 1.0)

        sample = {
            'input': torch.from_numpy(inp),
            'target': torch.from_numpy(vox)
        }
        return sample


class ConvBlock3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv3d(in_ch, out_ch, k, padding=p)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, k, padding=p)
        self.bn2 = nn.BatchNorm3d(out_ch)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class UNet3D(nn.Module):
    def __init__(self, in_ch: int = 1, base_ch: int = 16):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock3d(in_ch, base_ch)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3d(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock3d(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = ConvBlock3d(base_ch * 4, base_ch * 8)

        # Decoder
        self.up3 = nn.ConvTranspose3d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = ConvBlock3d(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock3d(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose3d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock3d(base_ch * 2, base_ch)

        self.out_conv = nn.Conv3d(base_ch, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        logits = self.out_conv(d1)
        return logits


def bce_with_logits_loss(pred: torch.Tensor, target: torch.Tensor, pos_weight: Optional[float] = 4.0):
    if pos_weight is not None:
        pw = torch.tensor([pos_weight], device=pred.device, dtype=pred.dtype)
        return F.binary_cross_entropy_with_logits(pred, target, pos_weight=pw)
    return F.binary_cross_entropy_with_logits(pred, target)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    ds = RadarDataset(
        data_dir=args.data_dir,
        pattern=args.pattern,
        bounds=((args.xmin, args.xmax), (args.ymin, args.ymax), (args.zmin, args.zmax)),
        grid_size=(args.nx, args.ny, args.nz),
        augment=True
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = UNet3D(in_ch=1, base_ch=args.base_ch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_loss = math.inf
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in dl:
            x = batch['input'].to(device, dtype=torch.float32)
            y = batch['target'].to(device, dtype=torch.float32)
            opt.zero_grad()
            logits = model(x)
            loss = bce_with_logits_loss(logits, y, pos_weight=4.0)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)

        epoch_loss = running / len(ds)
        print(f"Epoch {epoch:03d} | loss={epoch_loss:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(args.out_dir, f"unet3d_epoch{epoch:03d}.pt")
        torch.save({'epoch': epoch, 'model': model.state_dict()}, ckpt_path)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = os.path.join(args.out_dir, "unet3d_best.pt")
            torch.save({'epoch': epoch, 'model': model.state_dict()}, best_path)
            print(f"Saved best: {best_path}")


def infer(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model = UNet3D(in_ch=1, base_ch=args.base_ch).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Load a single CSV or a directory
    files = []
    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, args.pattern)))
        if not files:
            raise ValueError(f"No files found in {args.input} matching {args.pattern}")
    else:
        files = [args.input]

    bounds = ((args.xmin, args.xmax), (args.ymin, args.ymax), (args.zmin, args.zmax))
    grid = (args.nx, args.ny, args.nz)

    out_dir = args.out_dir or "dl_outputs"
    os.makedirs(out_dir, exist_ok=True)

    for fpath in files:
        data = np.genfromtxt(fpath, delimiter=',', skip_header=1)
        pts = data[:, :3].astype(np.float32) if data.size else np.zeros((0, 3), dtype=np.float32)
        vox_in = points_to_voxels(pts, bounds, grid)
        x = torch.from_numpy(vox_in).to(device, dtype=torch.float32)
        with torch.no_grad():
            logits = model(x)
            prob = torch.sigmoid(logits).cpu().numpy()[0, 0]  # (Z,Y,X)

        # Threshold and visualize occupied voxels as points
        occ = prob > args.thresh
        iz, iy, ix = np.where(occ)
        if iz.size == 0:
            print(f"No occupied voxels for {Path(fpath).name} at threshold {args.thresh}")
            continue

        # Convert voxel indices back to metric coordinates (voxel centers)
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
        nx, ny, nz = grid
        cx = xmin + (ix + 0.5) * (xmax - xmin) / nx
        cy = ymin + (iy + 0.5) * (ymax - ymin) / ny
        cz = zmin + (iz + 0.5) * (zmax - zmin) / nz

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(cx, cy, cz, s=4, c=cz, cmap='viridis', alpha=0.7)
        plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.1, label='Z (m)')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('DL 3D Reconstruction (occupied voxels)')
        ax.view_init(elev=30, azim=45)
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"recon_{Path(fpath).stem}.png")
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved {out_path}")


def build_argparser():
    p = argparse.ArgumentParser(description='Deep Learning 3D Reconstruction from Radar Point Clouds')
    sub = p.add_subparsers(dest='cmd', required=True)

    # Train
    pt = sub.add_parser('train', help='Train UNet3D on voxelized frames')
    pt.add_argument('--data-dir', type=str, default='csv_sample_data')
    pt.add_argument('--pattern', type=str, default='pointcloud_frame_*.csv')
    pt.add_argument('--nx', type=int, default=128)
    pt.add_argument('--ny', type=int, default=128)
    pt.add_argument('--nz', type=int, default=32)
    pt.add_argument('--xmin', type=float, default=-50.0)
    pt.add_argument('--xmax', type=float, default=50.0)
    pt.add_argument('--ymin', type=float, default=-50.0)
    pt.add_argument('--ymax', type=float, default=50.0)
    pt.add_argument('--zmin', type=float, default=-5.0)
    pt.add_argument('--zmax', type=float, default=5.0)
    pt.add_argument('--epochs', type=int, default=10)
    pt.add_argument('--batch-size', type=int, default=2)
    pt.add_argument('--lr', type=float, default=1e-3)
    pt.add_argument('--base-ch', type=int, default=16)
    pt.add_argument('--out-dir', type=str, default='checkpoints')
    pt.add_argument('--cpu', action='store_true')

    # Infer
    pi = sub.add_parser('infer', help='Infer DL reconstruction and save PNG')
    pi.add_argument('--input', type=str, default='sample_data')
    pi.add_argument('--pattern', type=str, default='pointcloud_frame_*.csv')
    pi.add_argument('--checkpoint', type=str, required=True)
    pi.add_argument('--nx', type=int, default=128)
    pi.add_argument('--ny', type=int, default=128)
    pi.add_argument('--nz', type=int, default=32)
    pi.add_argument('--xmin', type=float, default=-50.0)
    pi.add_argument('--xmax', type=float, default=50.0)
    pi.add_argument('--ymin', type=float, default=-50.0)
    pi.add_argument('--ymax', type=float, default=50.0)
    pi.add_argument('--zmin', type=float, default=-5.0)
    pi.add_argument('--zmax', type=float, default=5.0)
    pi.add_argument('--base-ch', type=int, default=16)
    pi.add_argument('--thresh', type=float, default=0.4)
    pi.add_argument('--out-dir', type=str, default='dl_outputs')
    pi.add_argument('--cpu', action='store_true')

    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()
    if args.cmd == 'train':
        train(args)
    elif args.cmd == 'infer':
        infer(args)
    else:
        raise ValueError(f"Unknown cmd {args.cmd}")


if __name__ == '__main__':
    main()



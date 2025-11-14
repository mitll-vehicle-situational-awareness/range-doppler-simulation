"""
Radar Point Cloud 3D Reconstruction and Visualization
Takes radar data (CSV or MAT files) and creates a 3D reconstruction of the environment.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
from pathlib import Path
try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available. Cannot load .mat files.")


class Radar3DReconstruction:
    """Class for 3D reconstruction and visualization of radar point cloud data."""
    
    def __init__(self):
        self.points = None
        self.intensity = None
        self.velocity = None
        self.timestamps = None
        
    def load_from_csv(self, csv_path):
        """
        Load point cloud data from a single CSV file.
        
        Args:
            csv_path: Path to CSV file with columns: X, Y, Z, Intensity, Vx, Vy, Vz
        """
        data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
        if data.size == 0:
            print(f"Warning: No data found in {csv_path}")
            return
        
        self.points = data[:, :3]  # X, Y, Z
        self.intensity = data[:, 3] if data.shape[1] > 3 else None
        if data.shape[1] > 6:
            self.velocity = data[:, 4:7]  # Vx, Vy, Vz
        else:
            self.velocity = None
            
        return self
    
    def load_multiple_csv_frames(self, data_dir, pattern="pointcloud_frame_*.csv"):
        """
        Load multiple CSV frame files and aggregate them into a single point cloud.
        
        Args:
            data_dir: Directory containing CSV files
            pattern: File pattern to match
        """
        csv_files = sorted(glob.glob(os.path.join(data_dir, pattern)))
        
        if not csv_files:
            raise ValueError(f"No CSV files found matching pattern: {pattern}")
        
        print(f"Loading {len(csv_files)} CSV frame files...")
        
        all_points = []
        all_intensity = []
        all_velocity = []
        frame_numbers = []
        
        for i, csv_file in enumerate(csv_files):
            try:
                data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
                if data.size == 0:
                    continue
                    
                all_points.append(data[:, :3])
                if data.shape[1] > 3:
                    all_intensity.append(data[:, 3])
                if data.shape[1] > 6:
                    all_velocity.append(data[:, 4:7])
                frame_numbers.append(i)
                
            except Exception as e:
                print(f"Warning: Error loading {csv_file}: {e}")
                continue
        
        if not all_points:
            raise ValueError("No valid data loaded from CSV files")
        
        # Aggregate all points
        self.points = np.vstack(all_points)
        self.intensity = np.hstack(all_intensity) if all_intensity else None
        self.velocity = np.vstack(all_velocity) if all_velocity else None
        self.timestamps = np.array(frame_numbers)
        
        print(f"Aggregated {len(self.points)} points from {len(frame_numbers)} frames")
        return self
    
    def load_from_mat(self, mat_path, point_cloud_key='allPointClouds'):
        """
        Load point cloud data from a MATLAB .mat file.
        
        Args:
            mat_path: Path to .mat file
            point_cloud_key: Key name for point cloud data in .mat file
        """
        if not HAS_SCIPY:
            raise ImportError("scipy is required to load .mat files. Install with: pip install scipy")
        
        print(f"Loading .mat file: {mat_path}")
        data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        
        if point_cloud_key not in data:
            # Try to find point cloud data
            available_keys = [k for k in data.keys() if not k.startswith('__')]
            print(f"Available keys: {available_keys}")
            if available_keys:
                point_cloud_key = available_keys[0]
                print(f"Using key: {point_cloud_key}")
            else:
                raise KeyError(f"Point cloud data not found in .mat file. Available keys: {list(data.keys())}")
        
        point_clouds = data[point_cloud_key]
        
        # Handle different .mat file structures
        if isinstance(point_clouds, np.ndarray):
            if point_clouds.dtype.names is not None:
                # Structured array
                all_points = []
                all_intensity = []
                all_velocity = []
                
                for pc in point_clouds.flat:
                    if hasattr(pc, 'Location') or 'Location' in pc.dtype.names:
                        loc = pc.Location if hasattr(pc, 'Location') else pc['Location']
                        all_points.append(loc)
                        
                        if hasattr(pc, 'Intensity') or 'Intensity' in pc.dtype.names:
                            intensity = pc.Intensity if hasattr(pc, 'Intensity') else pc['Intensity']
                            all_intensity.append(intensity)
                        
                        if hasattr(pc, 'Velocity') or 'Velocity' in pc.dtype.names:
                            vel = pc.Velocity if hasattr(pc, 'Velocity') else pc['Velocity']
                            all_velocity.append(vel)
            else:
                # Regular array - try to extract points
                all_points = [point_clouds]
        else:
            # List or other structure
            all_points = []
            all_intensity = []
            all_velocity = []
            
            # Handle list of structs (cell array in MATLAB)
            for pc in point_clouds:
                if isinstance(pc, np.ndarray) and pc.dtype.names:
                    if 'Location' in pc.dtype.names:
                        all_points.append(pc['Location'])
                    if 'Intensity' in pc.dtype.names:
                        all_intensity.append(pc['Intensity'])
                    if 'Velocity' in pc.dtype.names:
                        all_velocity.append(pc['Velocity'])
                elif isinstance(pc, dict):
                    if 'Location' in pc:
                        all_points.append(pc['Location'])
                    if 'Intensity' in pc:
                        all_intensity.append(pc['Intensity'])
                    if 'Velocity' in pc:
                        all_velocity.append(pc['Velocity'])
        
        # Aggregate points
        if all_points:
            self.points = np.vstack(all_points) if len(all_points) > 1 else all_points[0]
            self.intensity = np.hstack(all_intensity) if all_intensity else None
            self.velocity = np.vstack(all_velocity) if all_velocity else None
        else:
            raise ValueError("Could not extract point cloud data from .mat file")
        
        print(f"Loaded {len(self.points)} points from .mat file")
        return self
    
    def filter_points(self, range_max=None, z_min=None, z_max=None):
        """
        Filter points based on criteria.
        
        Args:
            range_max: Maximum distance from origin
            z_min: Minimum Z value
            z_max: Maximum Z value
        """
        if self.points is None:
            raise ValueError("No point cloud data loaded")
        
        mask = np.ones(len(self.points), dtype=bool)
        
        if range_max is not None:
            ranges = np.linalg.norm(self.points, axis=1)
            mask &= (ranges <= range_max)
        
        if z_min is not None:
            mask &= (self.points[:, 2] >= z_min)
        
        if z_max is not None:
            mask &= (self.points[:, 2] <= z_max)
        
        self.points = self.points[mask]
        if self.intensity is not None:
            self.intensity = self.intensity[mask]
        if self.velocity is not None:
            self.velocity = self.velocity[mask]
        
        print(f"Filtered to {len(self.points)} points")
        return self
    
    def visualize_3d(self, use_intensity=True, use_velocity=False, 
                     point_size=10, alpha=0.6, colormap='viridis',
                     save_path=None, show=True):
        """
        Create an interactive 3D visualization of the point cloud.
        
        Args:
            use_intensity: Color points by intensity (SNR)
            use_velocity: Color points by velocity magnitude
            point_size: Size of points in plot
            alpha: Transparency (0-1)
            colormap: Matplotlib colormap name
            save_path: Optional path to save figure
            show: Whether to display the plot
        """
        if self.points is None or len(self.points) == 0:
            raise ValueError("No point cloud data to visualize")
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Determine color scheme
        if use_velocity and self.velocity is not None:
            # Color by velocity magnitude
            velocity_mag = np.linalg.norm(self.velocity, axis=1)
            colors = velocity_mag
            color_label = 'Velocity Magnitude (m/s)'
        elif use_intensity and self.intensity is not None:
            # Color by intensity (SNR)
            colors = self.intensity
            color_label = 'Intensity (SNR)'
        else:
            # Color by height (Z)
            colors = self.points[:, 2]
            color_label = 'Height (Z, m)'
        
        # Create scatter plot
        scatter = ax.scatter(self.points[:, 0], 
                            self.points[:, 1], 
                            self.points[:, 2],
                            c=colors,
                            s=point_size,
                            alpha=alpha,
                            cmap=colormap)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.1)
        cbar.set_label(color_label, rotation=270, labelpad=20)
        
        # Set labels and title
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title('3D Radar Point Cloud Reconstruction', fontsize=14, fontweight='bold')
        
        # Set equal aspect ratio
        max_range = np.array([
            self.points[:, 0].max() - self.points[:, 0].min(),
            self.points[:, 1].max() - self.points[:, 1].min(),
            self.points[:, 2].max() - self.points[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (self.points[:, 0].max() + self.points[:, 0].min()) * 0.5
        mid_y = (self.points[:, 1].max() + self.points[:, 1].min()) * 0.5
        mid_z = (self.points[:, 2].max() + self.points[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Set initial view angle (top-down perspective)
        ax.view_init(elev=30, azim=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            return fig, ax
    
    def visualize_multiple_views(self, save_dir=None):
        """
        Create multiple view angles of the 3D reconstruction.
        
        Args:
            save_dir: Directory to save view images
        """
        views = [
            ('top_down', {'elev': 90, 'azim': 0}),
            ('side', {'elev': 0, 'azim': 0}),
            ('isometric', {'elev': 30, 'azim': 45}),
            ('front', {'elev': 0, 'azim': 90})
        ]
        
        for view_name, view_params in views:
            fig, ax = self.visualize_3d(show=False)
            ax.view_init(**view_params)
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'3d_reconstruction_{view_name}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved {view_name} view to: {save_path}")
            else:
                plt.show()
            
            plt.close()


def main():
    """Main function to demonstrate radar 3D reconstruction."""
    import argparse
    
    parser = argparse.ArgumentParser(description='3D Radar Point Cloud Reconstruction')
    parser.add_argument('--input', type=str, default='sample_data',
                       help='Input directory (for CSV) or file path (for CSV/MAT)')
    parser.add_argument('--format', type=str, choices=['csv', 'mat', 'auto'],
                       default='auto', help='Input file format')
    parser.add_argument('--range-max', type=float, default=None,
                       help='Maximum range to display (meters)')
    parser.add_argument('--z-min', type=float, default=None,
                       help='Minimum Z value')
    parser.add_argument('--z-max', type=float, default=None,
                       help='Maximum Z value')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save visualization image')
    parser.add_argument('--multi-view', action='store_true',
                       help='Generate multiple view angles')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save multiple views')
    
    args = parser.parse_args()
    
    # Create reconstruction object
    recon = Radar3DReconstruction()
    
    # Determine input format
    input_path = args.input
    
    if args.format == 'auto':
        if os.path.isdir(input_path):
            args.format = 'csv'
        elif input_path.endswith('.mat'):
            args.format = 'mat'
        elif input_path.endswith('.csv'):
            args.format = 'csv'
        else:
            args.format = 'csv'
    
    # Load data
    try:
        if args.format == 'csv':
            if os.path.isdir(input_path):
                recon.load_multiple_csv_frames(input_path)
            else:
                recon.load_from_csv(input_path)
        elif args.format == 'mat':
            recon.load_from_mat(input_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Filter points if requested
    if args.range_max or args.z_min or args.z_max:
        recon.filter_points(range_max=args.range_max,
                           z_min=args.z_min,
                           z_max=args.z_max)
    
    # Visualize
    if args.multi_view:
        recon.visualize_multiple_views(save_dir=args.save_dir)
    else:
        recon.visualize_3d(save_path=args.save, show=True)


if __name__ == '__main__':
    main()


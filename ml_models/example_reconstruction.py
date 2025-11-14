"""
Example script demonstrating 3D radar point cloud reconstruction.

This script shows how to use the Radar3DReconstruction class to:
1. Load radar data from CSV files or MAT files
2. Filter and process the point cloud
3. Visualize the 3D reconstruction
"""

from models.radar import Radar3DReconstruction
import os

def example_csv_reconstruction():
    """Example: Load and visualize from CSV files."""
    print("=" * 60)
    print("Example 1: Loading from CSV files")
    print("=" * 60)
    
    # Create reconstruction object
    recon = Radar3DReconstruction()
    
    # Load all CSV frames from sample_data directory
    csv_dir = 'sample_data'
    if os.path.exists(csv_dir):
        recon.load_multiple_csv_frames(csv_dir)
        
        # Filter points (optional)
        # Remove points beyond 100m range
        recon.filter_points(range_max=100.0)
        
        # Visualize
        print("\nDisplaying 3D reconstruction...")
        recon.visualize_3d(
            use_intensity=True,
            point_size=15,
            alpha=0.7,
            save_path='3d_reconstruction_output.png'
        )
    else:
        print(f"Directory {csv_dir} not found. Skipping CSV example.")


def example_mat_reconstruction():
    """Example: Load and visualize from MAT file."""
    print("\n" + "=" * 60)
    print("Example 2: Loading from MAT file")
    print("=" * 60)
    
    # Create reconstruction object
    recon = Radar3DReconstruction()
    
    # Load from MAT file
    mat_file = 'radar_pointcloud_data.mat'
    if os.path.exists(mat_file):
        try:
            recon.load_from_mat(mat_file)
            
            # Filter points (optional)
            recon.filter_points(range_max=100.0, z_max=10.0)
            
            # Visualize with velocity coloring
            print("\nDisplaying 3D reconstruction colored by velocity...")
            recon.visualize_3d(
                use_intensity=False,
                use_velocity=True,
                point_size=20,
                alpha=0.8,
                colormap='plasma',
                save_path='3d_reconstruction_mat_output.png'
            )
        except Exception as e:
            print(f"Error loading MAT file: {e}")
    else:
        print(f"File {mat_file} not found. Skipping MAT example.")


def example_multiple_views():
    """Example: Generate multiple view angles."""
    print("\n" + "=" * 60)
    print("Example 3: Multiple view angles")
    print("=" * 60)
    
    # Create reconstruction object
    recon = Radar3DReconstruction()
    
    # Load data
    csv_dir = 'sample_data'
    if os.path.exists(csv_dir):
        recon.load_multiple_csv_frames(csv_dir)
        recon.filter_points(range_max=100.0)
        
        # Generate multiple views
        output_dir = 'reconstruction_views'
        print(f"\nGenerating multiple views in {output_dir}...")
        recon.visualize_multiple_views(save_dir=output_dir)
    else:
        print(f"Directory {csv_dir} not found. Skipping multi-view example.")


if __name__ == '__main__':
    # Run examples
    example_csv_reconstruction()
    # example_mat_reconstruction()  # Uncomment if you have MAT files
    # example_multiple_views()  # Uncomment to generate multiple views
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


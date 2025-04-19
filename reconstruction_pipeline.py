import os
import numpy as np
import torch
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
import time
import rembg

# Import our modules - make sure the files are in the same directory
from image_preprocessing import load_image, remove_background, resize_foreground
from depth_estimation import DepthEstimator
from depth_to_pointcloud import create_point_cloud_from_depth, save_point_cloud
from point_cloud_processing import preprocess_point_cloud, remove_outliers
from surface_reconstruction import poisson_surface_reconstruction, save_mesh, visualize_mesh

class Timer:
    """Simple timer for performance measurements"""
    def __init__(self):
        self.timings = {}
        self._current_section = None
        self._start_time = None
    
    def start(self, section_name):
        """Start timing a section"""
        if self._current_section is not None:
            self.end(self._current_section)
        
        self._current_section = section_name
        self._start_time = time.time()
        print(f"Starting: {section_name}")
    
    def end(self, section_name=None):
        """End timing a section"""
        if section_name is None:
            section_name = self._current_section
        
        if self._start_time is not None:
            elapsed = time.time() - self._start_time
            self.timings[section_name] = elapsed
            print(f"Completed: {section_name} in {elapsed:.2f} seconds")
            
            self._current_section = None
            self._start_time = None
    
    def summary(self):
        """Print timing summary"""
        print("\n--- Timing Summary ---")
        total_time = sum(self.timings.values())
        for section, elapsed in self.timings.items():
            print(f"{section}: {elapsed:.2f}s ({elapsed/total_time*100:.1f}%)")
        print(f"Total time: {total_time:.2f}s")

def reconstruct_3d_from_image(image_path, output_dir="output", visualize=True):
    """Complete pipeline to reconstruct 3D from a single image"""
    timer = Timer()
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depth_maps"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "point_clouds"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "meshes"), exist_ok=True)
    
    # Get filename without extension for outputs
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
        # 1. Load and preprocess image
    timer.start("Image Loading & Preprocessing")
    original_image = load_image(image_path)
    rembg_session = rembg.new_session()
    image_no_bg = remove_background(original_image, rembg_session)

    # Create a processed image for depth estimation (ensuring RGB format)
    processed_image = resize_foreground(image_no_bg, target_size=(512, 512))
    if processed_image.mode == "RGBA":
        # Create a white background for the RGB image
        rgb_image = Image.new("RGB", processed_image.size, (255, 255, 255))
        # Paste the image with transparency
        rgb_image.paste(processed_image, mask=processed_image.split()[3])
        processed_image_rgb = rgb_image
    else:
        processed_image_rgb = processed_image.convert("RGB")
    timer.end()

    # 2. Estimate depth
    timer.start("Depth Estimation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth_estimator = DepthEstimator(device=device)
    normalized_depth, depth_map = depth_estimator.estimate_depth(processed_image_rgb)

    
    # Save depth map
    depth_map_path = os.path.join(output_dir, "depth_maps", f"{base_filename}_depth.png")
    depth_estimator.visualize_depth(normalized_depth, depth_map_path)
    timer.end()
    
    # 3. Create point cloud from depth
    timer.start("Point Cloud Generation")
    point_cloud = create_point_cloud_from_depth(processed_image, depth_map)
    
    # Save raw point cloud
    raw_pcd_path = os.path.join(output_dir, "point_clouds", f"{base_filename}_raw.ply")
    save_point_cloud(point_cloud, raw_pcd_path)
    timer.end()
    
    # 4. Process point cloud
    timer.start("Point Cloud Processing")
    # Estimate appropriate voxel size based on point cloud density
    points = np.asarray(point_cloud.points)
    bbox = point_cloud.get_axis_aligned_bounding_box()
    bbox_size = bbox.get_extent()
    voxel_size = min(bbox_size) / 100  # Adaptive voxel size
    
    processed_pcd = preprocess_point_cloud(point_cloud, voxel_size=voxel_size)
    filtered_pcd = remove_outliers(processed_pcd)
    
    # Save processed point cloud
    processed_pcd_path = os.path.join(output_dir, "point_clouds", f"{base_filename}_processed.ply")
    save_point_cloud(filtered_pcd, processed_pcd_path)
    timer.end()
    
    # 5. Surface reconstruction
    timer.start("Surface Reconstruction")
    # Determine appropriate octree depth based on point cloud size
    num_points = len(filtered_pcd.points)
    octree_depth = min(10, max(6, int(np.log2(num_points) - 6)))
    
    mesh = poisson_surface_reconstruction(filtered_pcd, depth=octree_depth)
    
    # Save mesh
    mesh_path = os.path.join(output_dir, "meshes", f"{base_filename}_mesh.obj")
    save_mesh(mesh, mesh_path)
    timer.end()
    
    # 6. Visualization
    if visualize:
        print("\nVisualization results:")
        
        # Show original image, depth map, and reconstructed model side by side
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(np.array(original_image))
        plt.title("Original Image")
        plt.axis("off")
        
        plt.subplot(1, 3, 2)
        plt.imshow(normalized_depth, cmap="plasma")
        plt.title("Depth Map")
        plt.axis("off")
        
        plt.subplot(1, 3, 3)
        plt.text(0.5, 0.5, "3D Model\n(See separate window)", 
                 ha="center", va="center", fontsize=12)
        plt.axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_filename}_results.png"))
        plt.show()
        
        # Show 3D visualizations
        visualize_mesh(mesh, window_name=f"3D Reconstruction: {base_filename}")
    
    # Summary
    timer.summary()
    
    return {
        "depth_map_path": depth_map_path,
        "point_cloud_path": processed_pcd_path,
        "mesh_path": mesh_path,
        "summary_image": os.path.join(output_dir, f"{base_filename}_results.png")
    }

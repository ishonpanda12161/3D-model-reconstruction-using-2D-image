import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image
import torch

def create_point_cloud_from_depth(rgb_image, depth_map, fx=None, fy=None, cx=None, cy=None):
    """
    Convert RGB image and depth map to colored point cloud
    
    Parameters:
    - rgb_image: PIL Image or numpy array with RGB values
    - depth_map: Numpy array with depth values
    - fx, fy: Focal length of the camera (if None, estimated from image size)
    - cx, cy: Principal point of the camera (if None, estimated as image center)
    
    Returns:
    - point_cloud: Open3D PointCloud object
    """
    # Convert PIL image to numpy if needed
    if isinstance(rgb_image, Image.Image):
        # Ensure we're working with an RGB image
        if rgb_image.mode == "RGBA":
            # Remove alpha channel
            background = Image.new("RGB", rgb_image.size, (255, 255, 255))
            background.paste(rgb_image, mask=rgb_image.split()[3])
            rgb_image = background
        elif rgb_image.mode != "RGB":
            rgb_image = rgb_image.convert("RGB")
        rgb = np.array(rgb_image)
    else:
        rgb = rgb_image
    
    # Get image dimensions
    height, width = depth_map.shape
    
    # Estimate camera parameters if not provided
    if fx is None or fy is None:
        # Estimate focal length (assume 60° field of view)
        fov = 60
        fx = width / (2 * np.tan(np.radians(fov / 2)))
        fy = height / (2 * np.tan(np.radians(fov / 2)))
    
    if cx is None or cy is None:
        # Estimate principal point as image center
        cx = width / 2
        cy = height / 2
    
    # Create meshgrid for pixel coordinates
    v, u = np.mgrid[0:height, 0:width]
    
    # Get valid depth points (non-zero and non-NaN)
    valid_depth = np.logical_and(depth_map > 0, ~np.isnan(depth_map))
    u_valid = u[valid_depth]
    v_valid = v[valid_depth]
    z_valid = depth_map[valid_depth]
    
    # Calculate 3D coordinates
    x = (u_valid - cx) * z_valid / fx
    y = (v_valid - cy) * z_valid / fy
    z = z_valid
    
    # Stack coordinates
    points = np.stack((x, y, z), axis=1)
    
    # Get colors for valid points - ensure RGB format
    if len(rgb.shape) == 3:  # Color image (RGB/RGBA)
        if rgb.shape[2] == 4:  # RGBA format
            rgb = rgb[:, :, :3]  # Remove alpha channel
        colors = rgb[v_valid, u_valid] / 255.0  # Normalize to [0, 1]
    else:  # Grayscale image
        # Convert to RGB
        colors = np.repeat(rgb[v_valid, u_valid][:, np.newaxis] / 255.0, 3, axis=1)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def visualize_point_cloud(point_cloud, window_name="Point Cloud Visualization"):
    """Visualize point cloud using Open3D"""
    # Create coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    
    # Visualize
    o3d.visualization.draw_geometries([point_cloud, coordinate_frame], 
                                      window_name=window_name,
                                      width=800, height=600)

def save_point_cloud(point_cloud, output_path):
    """Save point cloud to file"""
    o3d.io.write_point_cloud(output_path, point_cloud)
    print(f"Point cloud saved to: {output_path}")

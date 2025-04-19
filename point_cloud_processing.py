import numpy as np
import open3d as o3d

def preprocess_point_cloud(pcd, voxel_size=0.02):
    """Downsample and normalize point cloud"""
    print("Preprocessing point cloud...")
    
    # Downsample the point cloud
    print(f"Downsampling with voxel size: {voxel_size}")
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # Estimate normals if they don't exist
    if not pcd_down.has_normals():
        print("Estimating normals...")
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
        pcd_down.orient_normals_towards_camera_location()
    
    return pcd_down

def remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    """Remove outliers from point cloud"""
    print("Removing outliers...")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return cl

def estimate_point_cloud_density(pcd):
    """Estimate point cloud density for reconstruction parameters"""
    # Compute nearest neighbor distance for each point
    distances = pcd.compute_nearest_neighbor_distance()
    avg_distance = np.mean(distances)
    print(f"Average nearest neighbor distance: {avg_distance:.6f}")
    return avg_distance

import numpy as np
import open3d as o3d

def poisson_surface_reconstruction(pcd, depth=9, scale=1.1, linear_fit=False):
    """
    Perform Poisson surface reconstruction
    
    Parameters:
    - pcd: Open3D point cloud with normals
    - depth: Maximum depth of the octree used for reconstruction
    - scale: Scale factor for reconstructing the mesh
    - linear_fit: Use linear interpolation to fit triangles
    
    Returns:
    - mesh: Reconstructed triangle mesh
    """
    print(f"Performing Poisson surface reconstruction with depth={depth}...")
    
    # Ensure we have normals
    if not pcd.has_normals():
        pcd.estimate_normals()
        pcd.orient_normals_towards_camera_location()
    
    # Perform reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=scale, linear_fit=linear_fit)
    
    # Calculate density threshold for removing low-density vertices
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.1)  # Remove bottom 10% density
    print(f"Density threshold: {density_threshold}")
    
    # Filter mesh based on vertex density
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    print(f"Reconstruction complete. Mesh has {len(mesh.triangles)} triangles.")
    return mesh

def visualize_mesh(mesh, window_name="Mesh Visualization"):
    """Visualize mesh using Open3D"""
    # Create coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    
    # Visualize
    o3d.visualization.draw_geometries([mesh, coordinate_frame], 
                                     window_name=window_name,
                                     width=800, height=600,
                                     mesh_show_back_face=True)

def save_mesh(mesh, output_path):
    """Save mesh to file"""
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"Mesh saved to: {output_path}")

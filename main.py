import os
import argparse
from reconstruction_pipeline import reconstruct_3d_from_image

def parse_arguments():
    parser = argparse.ArgumentParser(description="3D Reconstruction from 2D Images")
    parser.add_argument("--input", type=str, required=True, 
                        help="Input image path")
    parser.add_argument("--output", type=str, default="output",
                        help="Output directory for results")
    parser.add_argument("--visualize", action="store_true",
                        help="Show visualizations during processing")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Check if input image exists
    if not os.path.isfile(args.input):
        print(f"Error: Input image {args.input} does not exist")
        return
    
    # Run the reconstruction pipeline
    result = reconstruct_3d_from_image(
        image_path=args.input,
        output_dir=args.output,
        visualize=args.visualize
    )
    
    print("\nReconstruction results:")
    print(f"Depth map: {result['depth_map_path']}")
    print(f"Point cloud: {result['point_cloud_path']}")
    print(f"3D mesh: {result['mesh_path']}")
    print(f"Summary image: {result['summary_image']}")

if __name__ == "__main__":
    main()

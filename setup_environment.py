# Create this file first to check your CUDA setup
import os
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# Create directories
os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs("output/depth_maps", exist_ok=True)
os.makedirs("output/point_clouds", exist_ok=True)
os.makedirs("output/meshes", exist_ok=True)

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("Using CPU for processing")

print("Environment setup complete!")

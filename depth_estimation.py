# depth_estimation.py
import torch
from transformers import DPTForDepthEstimation, DPTImageProcessor
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Debug print to verify module loading
print("Loading depth_estimation.py module")

class DepthEstimator:
    def __init__(self, model_name="Intel/dpt-hybrid-midas", device=None):
        """Initialize depth estimation model"""
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Loading depth estimation model: {model_name}")
        self.processor = DPTImageProcessor.from_pretrained(model_name)
        self.model = DPTForDepthEstimation.from_pretrained(model_name)
        self.model.to(self.device)
        print("Model loaded successfully")
        
    def estimate_depth(self, image):
        """Estimate depth from an image"""
        # Ensure the image is in RGB format (not RGBA)
        if hasattr(image, 'mode') and image.mode == "RGBA":
            # Convert RGBA to RGB by compositing with white background
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])  # Use alpha channel as mask
            image = background
        elif hasattr(image, 'mode') and image.mode != "RGB":
            # Convert any non-RGB image to RGB
            image = image.convert("RGB")
        
        # Convert PIL Image to numpy array (what the model expects)
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
            
        # Preprocess the image
        inputs = self.processor(images=image_np, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Get image dimensions for resizing the depth map
        if isinstance(image, Image.Image):
            img_width, img_height = image.size
        else:
            img_height, img_width = image.shape[:2]
        
        # Convert the output to a depth map
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(img_height, img_width),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # Normalize depth map for visualization
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)
        
        return normalized_depth, depth_map
        
    def visualize_depth(self, depth_map, output_path=None):
        """Visualize and optionally save depth map"""
        plt.figure(figsize=(10, 10))
        plt.imshow(depth_map, cmap='plasma')
        plt.colorbar(label='Depth')
        plt.title('Estimated Depth Map')
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            print(f"Depth map saved to: {output_path}")
            
        plt.close()

# Debug print to verify the class is defined
print("DepthEstimator class defined successfully")

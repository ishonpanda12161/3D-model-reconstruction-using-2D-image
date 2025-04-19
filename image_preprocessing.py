import numpy as np
import cv2
from PIL import Image
import rembg
import torch
from torchvision import transforms

def load_image(image_path):
    """Load an image and convert to RGB"""
    image = Image.open(image_path).convert("RGB")
    return image

def remove_background(image, rembg_session=None):
    """Remove the background of an image using rembg"""
    if rembg_session is None:
        rembg_session = rembg.new_session()
    
    # Convert PIL image to numpy array if needed
    if isinstance(image, Image.Image):
        input_image = np.array(image)
    else:
        input_image = image
        
    # Remove background
    output_image = rembg.remove(input_image, session=rembg_session)
    return Image.fromarray(output_image)

def resize_foreground(image, target_size=(512, 512)):
    """Resize image while maintaining aspect ratio"""
    w, h = image.size
    ratio = min(target_size[0] / w, target_size[1] / h)
    new_size = (int(w * ratio), int(h * ratio))
    
    resized_image = image.resize(new_size, Image.LANCZOS)
    
    # Create a new image with the target size and paste the resized image
    new_image = Image.new("RGBA", target_size, (0, 0, 0, 0))
    new_image.paste(resized_image, ((target_size[0] - new_size[0]) // 2,
                                    (target_size[1] - new_size[1]) // 2))
    
    return new_image

def preprocess_for_depth_estimation(image, image_processor):
    """Preprocess image for depth estimation model"""
    if image.mode == "RGBA":
        # Blend alpha channel with white background
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image).convert("RGB")
    
    # Apply model-specific preprocessing
    inputs = image_processor(images=image, return_tensors="pt")
    return inputs

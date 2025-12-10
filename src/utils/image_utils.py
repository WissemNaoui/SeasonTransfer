"""Image loading and saving utilities."""
import numpy as np
import torch
from PIL import Image


def load_image(image_path, size=256):
    """Load an image and convert to tensor.
    
    Args:
        image_path: Path to image file
        size: Target size (will be resized to size x size)
    
    Returns:
        torch.Tensor: Image tensor of shape (1, 3, size, size) normalized to [-1, 1]
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    img_array = np.array(img).astype(np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    return tensor


def tensor_to_image(tensor):
    """Convert tensor to PIL Image.
    
    Args:
        tensor: torch.Tensor of shape (1, 3, H, W) with values in [-1, 1]
    
    Returns:
        PIL.Image: Image in RGB format
    """
    with torch.no_grad():
        img_array = (tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)


def save_image(tensor, save_path):
    """Save tensor as image file.
    
    Args:
        tensor: torch.Tensor of shape (1, 3, H, W) with values in [-1, 1]
        save_path: Path where to save the image
    """
    img = tensor_to_image(tensor)
    img.save(save_path)

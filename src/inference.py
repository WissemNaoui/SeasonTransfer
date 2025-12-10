"""
CycleGAN Inference Script.
Load trained generators and transform images.
"""
import torch
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys

from models.generator import Generator
from utils import config


def load_generator(checkpoint_path, device):
    """Load trained generator from checkpoint."""
    gen = Generator(img_channels=3, num_residuals=9).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gen.load_state_dict(checkpoint["state_dict"])
    gen.eval()
    print(f"✓ Generator loaded from {checkpoint_path}")
    return gen


def transform_image(image_path, generator, device):
    """
    Transform a single image using the generator.
    
    Args:
        image_path: Path to input image
        generator: Trained generator model
        device: torch device
    
    Returns:
        Transformed PIL Image
    """
    # Load and preprocess image
    image = np.array(Image.open(image_path).convert("RGB"))
    
    transform = A.Compose([
        A.Resize(width=256, height=256),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ])
    
    augmented = transform(image=image)
    input_tensor = augmented["image"].unsqueeze(0).to(device)
    
    # Generate output
    with torch.no_grad():
        output_tensor = generator(input_tensor)
    
    # Denormalize and convert to PIL
    output = output_tensor.squeeze(0).cpu().numpy()
    output = (output * 0.5 + 0.5) * 255  # [-1, 1] -> [0, 255]
    output = output.transpose(1, 2, 0).astype(np.uint8)
    
    return Image.fromarray(output)


def main():
    """Demo inference."""
    if len(sys.argv) < 3:
        print("Usage: python inference.py <checkpoint_path> <image_path> [output_path]")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    image_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "output.png"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load generator
    generator = load_generator(checkpoint_path, device)
    
    # Transform image
    print(f"Transforming {image_path}...")
    output_image = transform_image(image_path, generator, device)
    
    # Save result
    output_image.save(output_path)
    print(f"✓ Output saved to {output_path}")


if __name__ == "__main__":
    main()

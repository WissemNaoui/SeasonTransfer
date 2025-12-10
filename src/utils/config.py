"""
CycleGAN Training Configuration.
Centralized hyperparameters and data augmentation pipeline.
"""
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data paths (will be overridden in Colab notebook)
TRAIN_DIR = "/content/drive/MyDrive/SeasonsGAN/data/summer2winter_yosemite"
VAL_DIR = "val"

# Training hyperparameters
BATCH_SIZE = 1  # CycleGAN uses batch size 1 with Instance Normalization
LEARNING_RATE = 2e-4  # Adam with β1=0.5, β2=0.999
NUM_WORKERS = 4
NUM_EPOCHS = 200
LOAD_MODEL = False
SAVE_MODEL = True

# Loss weights
LAMBDA_IDENTITY = 0.0  # Identity loss weight (0.0 = disabled, 0.5 = enabled)
LAMBDA_CYCLE = 10  # Cycle consistency loss weight (critical!)

# Checkpoint filenames
CHECKPOINT_GEN_S = "gen_summer.pth.tar"  # Generator: Winter -> Summer
CHECKPOINT_GEN_W = "gen_winter.pth.tar"  # Generator: Summer -> Winter
CHECKPOINT_DISC_S = "disc_summer.pth.tar"  # Discriminator for Summer images
CHECKPOINT_DISC_W = "disc_winter.pth.tar"  # Discriminator for Winter images

# Image transformations
transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        # Normalize to [-1, 1] range (required for Tanh output)
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},  # Apply same transform to both images
)

# Validation transforms (no augmentation)
val_transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)


def print_config():
    """Print current configuration."""
    print("=" * 50)
    print("CycleGAN Configuration")
    print("=" * 50)
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Lambda Cycle: {LAMBDA_CYCLE}")
    print(f"Lambda Identity: {LAMBDA_IDENTITY}")
    print("=" * 50)


if __name__ == "__main__":
    print_config()

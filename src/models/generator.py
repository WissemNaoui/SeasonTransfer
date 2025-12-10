"""
ResNet-based Generator for CycleGAN.
Architecture: Encoder -> Transformer (ResNet blocks) -> Decoder
"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    ResNet block with reflection padding and instance normalization.
    Preserves spatial dimensions.
    """
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)  # Skip connection


class ConvBlock(nn.Module):
    """
    Convolutional block with optional downsampling/upsampling.
    """
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)


class Generator(nn.Module):
    """
    CycleGAN Generator with ResNet architecture.
    
    Architecture:
    - Initial conv: 7x7, 64 filters
    - Downsampling: 2 blocks (128, 256 filters)
    - Transformation: 9 ResNet blocks (256 filters)
    - Upsampling: 2 blocks (128, 64 filters)
    - Output conv: 7x7, 3 filters (RGB)
    
    Args:
        img_channels: Number of input/output channels (3 for RGB)
        num_features: Base number of features (64)
        num_residuals: Number of ResNet blocks (9 for 256x256 images)
    """
    
    def __init__(self, img_channels=3, num_features=64, num_residuals=9):
        super().__init__()
        
        # Initial convolution block
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=0, padding_mode="reflect"),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        
        # Downsampling (Encoder)
        self.down_blocks = nn.ModuleList([
            ConvBlock(num_features, num_features * 2, kernel_size=3, stride=2, padding=1),      # 64 -> 128
            ConvBlock(num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1),  # 128 -> 256
        ])
        
        # Transformation (ResNet blocks)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )
        
        # Upsampling (Decoder)
        self.up_blocks = nn.ModuleList([
            ConvBlock(num_features * 4, num_features * 2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),  # 256 -> 128
            ConvBlock(num_features * 2, num_features, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),      # 128 -> 64
        ])
        
        # Output layer
        self.last = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(num_features, img_channels, kernel_size=7, stride=1, padding=0),
            nn.Tanh(),  # Output range [-1, 1]
        )

    def forward(self, x):
        x = self.initial(x)
        
        for layer in self.down_blocks:
            x = layer(x)
        
        x = self.residual_blocks(x)
        
        for layer in self.up_blocks:
            x = layer(x)
        
        return self.last(x)


def test_generator():
    """Test generator with dummy input."""
    img_channels = 3
    img_size = 256
    x = torch.randn((2, img_channels, img_size, img_size))
    gen = Generator(img_channels, num_residuals=9)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {gen(x).shape}")
    assert gen(x).shape == x.shape, "Output shape mismatch!"
    print("✓ Generator test passed")


if __name__ == "__main__":
    test_generator()

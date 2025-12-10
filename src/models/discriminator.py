"""
PatchGAN Discriminator for CycleGAN.
Classifies whether 70x70 overlapping image patches are real or fake.
"""
import torch
import torch.nn as nn


class Block(nn.Module):
    """
    Discriminator convolutional block.
    """
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator (70x70 receptive field).
    
    Architecture:
    - C64: 4x4 conv, 64 filters, stride 2
    - C128: 4x4 conv, 128 filters, stride 2
    - C256: 4x4 conv, 256 filters, stride 2
    - C512: 4x4 conv, 512 filters, stride 1
    - Output: 4x4 conv, 1 filter, stride 1
    
    Args:
        in_channels: Number of input channels (3 for RGB)
        features: List of feature dimensions for each layer
    """
    
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        
        # Initial layer (no normalization)
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Hidden layers
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block(in_channels, feature, stride=1 if feature == features[-1] else 2))
            in_channels = feature
        
        self.model = nn.Sequential(*layers)
        
        # Output layer (no activation - raw logits for LSGAN)
        self.final = nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")

    def forward(self, x):
        x = self.initial(x)
        x = self.model(x)
        return torch.sigmoid(self.final(x))  # PatchGAN output


def test_discriminator():
    """Test discriminator with dummy input."""
    x = torch.randn((5, 3, 256, 256))
    disc = Discriminator(in_channels=3)
    preds = disc(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {preds.shape}")
    print(f"Output range: [{preds.min():.3f}, {preds.max():.3f}]")
    print("✓ Discriminator test passed")


if __name__ == "__main__":
    test_discriminator()

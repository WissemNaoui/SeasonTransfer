"""
Unit tests for Discriminator architecture.

Tests verify:
- Output shape is correct (PatchGAN output)
- Output values are in valid range [0, 1]
- Model can handle different batch sizes
"""
import torch
import pytest
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.discriminator import Discriminator


def test_discriminator_output_shape():
    """
    Test if the Discriminator outputs the correct PatchGAN shape.
    For 256x256 input, PatchGAN should output a smaller spatial map.
    """
    model = Discriminator(in_channels=3)
    model.eval()
    
    # Input: 256x256 image
    dummy_input = torch.randn(1, 3, 256, 256)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    # PatchGAN output should be smaller than input
    assert len(output.shape) == 4, "Output should be 4D tensor (B, C, H, W)"
    assert output.shape[0] == 1, "Batch size should match"
    assert output.shape[1] == 1, "Discriminator should output 1 channel"
    
    print(f"✅ Discriminator Shape Test Passed - Output shape: {output.shape}")


def test_discriminator_output_range():
    """
    Test that Discriminator output is in valid probability range [0, 1].
    This is enforced by the Sigmoid activation.
    """
    model = Discriminator(in_channels=3)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 256, 256)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    assert output.min() >= 0.0, f"Output min {output.min()} is below 0.0"
    assert output.max() <= 1.0, f"Output max {output.max()} is above 1.0"
    print("✅ Discriminator Output Range Test Passed")


def test_discriminator_batch_size():
    """
    Test that Discriminator can handle different batch sizes.
    """
    model = Discriminator(in_channels=3)
    model.eval()
    
    batch_sizes = [1, 2, 4]
    
    for bs in batch_sizes:
        dummy_input = torch.randn(bs, 3, 256, 256)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output.shape[0] == bs, f"Batch size mismatch for bs={bs}"
    
    print("✅ Discriminator Batch Size Test Passed")


def test_discriminator_no_nans():
    """
    Test that Discriminator does not produce NaN values.
    """
    model = Discriminator(in_channels=3)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 256, 256)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"
    print("✅ Discriminator NaN Test Passed")


if __name__ == "__main__":
    print("Running Discriminator Tests...")
    print("=" * 60)
    
    test_discriminator_output_shape()
    test_discriminator_output_range()
    test_discriminator_batch_size()
    test_discriminator_no_nans()
    
    print("=" * 60)
    print("✅ All Discriminator Tests Passed!")

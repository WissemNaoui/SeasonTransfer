"""
Unit tests for Generator architecture.

Tests verify:
- Output shape matches input shape
- No NaN values in output
- Model can handle different batch sizes
- Model works in eval mode
"""
import torch
import pytest
import sys
import os

# Add src to path so we can import the model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.generator import Generator


def test_generator_output_shape():
    """
    Test if the Generator accepts a 256x256 image and outputs the same shape.
    This is critical for CycleGAN - input and output must be identical dimensions.
    """
    # 1. Instantiate the model
    model = Generator(img_channels=3, num_residuals=9)
    model.eval()  # Set to evaluation mode
    
    # 2. Create a dummy input tensor (Batch Size 1, 3 Channels, 256 Height, 256 Width)
    dummy_input = torch.randn(1, 3, 256, 256)
    
    # 3. Run inference
    with torch.no_grad():
        output = model(dummy_input)
    
    # 4. Assertions
    assert output.shape == (1, 3, 256, 256), f"Output shape {output.shape} should match input shape (1, 3, 256, 256)"
    assert not torch.isnan(output).any(), "Output should not contain NaNs"
    print("✅ Generator Shape Test Passed")


def test_generator_no_nans():
    """
    Test that the Generator does not produce NaN values.
    NaNs indicate numerical instability or incorrect initialization.
    """
    model = Generator(img_channels=3, num_residuals=9)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 256, 256)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"
    print("✅ Generator NaN Test Passed")


def test_generator_batch_size():
    """
    Test that the Generator can handle different batch sizes.
    CycleGAN typically uses batch_size=1, but the model should be flexible.
    """
    model = Generator(img_channels=3, num_residuals=9)
    model.eval()
    
    batch_sizes = [1, 2, 4]
    
    for bs in batch_sizes:
        dummy_input = torch.randn(bs, 3, 256, 256)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output.shape == (bs, 3, 256, 256), f"Failed for batch size {bs}"
    
    print("✅ Generator Batch Size Test Passed")


def test_generator_output_range():
    """
    Test that the Generator output is in the expected range [-1, 1].
    This is enforced by the Tanh activation in the final layer.
    """
    model = Generator(img_channels=3, num_residuals=9)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 256, 256)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    assert output.min() >= -1.0, f"Output min {output.min()} is below -1.0"
    assert output.max() <= 1.0, f"Output max {output.max()} is above 1.0"
    print("✅ Generator Output Range Test Passed")


def test_generator_deterministic():
    """
    Test that the Generator produces the same output for the same input.
    This verifies there's no randomness in the forward pass.
    """
    model = Generator(img_channels=3, num_residuals=9)
    model.eval()
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    dummy_input = torch.randn(1, 3, 256, 256)
    
    with torch.no_grad():
        output1 = model(dummy_input)
        output2 = model(dummy_input)
    
    assert torch.allclose(output1, output2), "Generator should be deterministic in eval mode"
    print("✅ Generator Deterministic Test Passed")


if __name__ == "__main__":
    print("Running Generator Tests...")
    print("=" * 60)
    
    test_generator_output_shape()
    test_generator_no_nans()
    test_generator_batch_size()
    test_generator_output_range()
    test_generator_deterministic()
    
    print("=" * 60)
    print("✅ All Generator Tests Passed!")

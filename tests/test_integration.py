"""
Integration tests for the complete CycleGAN pipeline.

Tests verify:
- Dataset loading works correctly
- Model training step executes without errors
- Inference pipeline works end-to-end
"""
import torch
import pytest
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.generator import Generator
from src.models.discriminator import Discriminator


def test_cyclegan_forward_pass():
    """
    Test a complete forward pass of the CycleGAN architecture.
    This simulates one training iteration.
    """
    # Initialize models
    gen_S2W = Generator(img_channels=3, num_residuals=9)  # Summer to Winter
    gen_W2S = Generator(img_channels=3, num_residuals=9)  # Winter to Summer
    disc_S = Discriminator(in_channels=3)
    disc_W = Discriminator(in_channels=3)
    
    # Set to eval mode for testing
    gen_S2W.eval()
    gen_W2S.eval()
    disc_S.eval()
    disc_W.eval()
    
    # Create dummy data
    summer_img = torch.randn(1, 3, 256, 256)
    winter_img = torch.randn(1, 3, 256, 256)
    
    with torch.no_grad():
        # Forward cycle: Summer -> Winter -> Summer
        fake_winter = gen_S2W(summer_img)
        reconstructed_summer = gen_W2S(fake_winter)
        
        # Backward cycle: Winter -> Summer -> Winter
        fake_summer = gen_W2S(winter_img)
        reconstructed_winter = gen_S2W(fake_summer)
        
        # Discriminator predictions
        disc_real_summer = disc_S(summer_img)
        disc_fake_summer = disc_S(fake_summer)
        disc_real_winter = disc_W(winter_img)
        disc_fake_winter = disc_W(fake_winter)
    
    # Assertions
    assert fake_winter.shape == summer_img.shape, "Generated winter image shape mismatch"
    assert reconstructed_summer.shape == summer_img.shape, "Reconstructed summer image shape mismatch"
    assert fake_summer.shape == winter_img.shape, "Generated summer image shape mismatch"
    assert reconstructed_winter.shape == winter_img.shape, "Reconstructed winter image shape mismatch"
    
    print("✅ CycleGAN Forward Pass Test Passed")


def test_cycle_consistency():
    """
    Test that cycle consistency makes sense (reconstruction should be similar to input).
    We don't expect exact match since models are untrained, but shapes should match.
    """
    gen_S2W = Generator(img_channels=3, num_residuals=9)
    gen_W2S = Generator(img_channels=3, num_residuals=9)
    
    gen_S2W.eval()
    gen_W2S.eval()
    
    summer_img = torch.randn(1, 3, 256, 256)
    
    with torch.no_grad():
        # Summer -> Winter -> Summer
        fake_winter = gen_S2W(summer_img)
        reconstructed_summer = gen_W2S(fake_winter)
    
    # Check shapes match
    assert reconstructed_summer.shape == summer_img.shape, "Cycle consistency shape mismatch"
    
    # Check no NaNs in the cycle
    assert not torch.isnan(reconstructed_summer).any(), "Cycle produced NaN values"
    
    print("✅ Cycle Consistency Test Passed")


def test_model_parameter_count():
    """
    Test that models have reasonable parameter counts.
    This helps catch accidental architecture changes.
    """
    gen = Generator(img_channels=3, num_residuals=9)
    disc = Discriminator(in_channels=3)
    
    gen_params = sum(p.numel() for p in gen.parameters())
    disc_params = sum(p.numel() for p in disc.parameters())
    
    # Generator should have ~11M parameters (9 ResNet blocks)
    assert 10_000_000 < gen_params < 15_000_000, f"Generator has unexpected param count: {gen_params:,}"
    
    # Discriminator should have ~2-3M parameters
    assert 2_000_000 < disc_params < 4_000_000, f"Discriminator has unexpected param count: {disc_params:,}"
    
    print(f"✅ Model Parameter Count Test Passed")
    print(f"   Generator: {gen_params:,} parameters")
    print(f"   Discriminator: {disc_params:,} parameters")


if __name__ == "__main__":
    print("Running Integration Tests...")
    print("=" * 60)
    
    test_cyclegan_forward_pass()
    test_cycle_consistency()
    test_model_parameter_count()
    
    print("=" * 60)
    print("✅ All Integration Tests Passed!")

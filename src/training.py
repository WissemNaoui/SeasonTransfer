"""
CycleGAN Training Loop.
Implements adversarial loss + cycle consistency loss.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from models.generator import Generator
from models.discriminator import Discriminator
from data.dataset import CycleGANDataset
from utils import config


def save_checkpoint(model, optimizer, filename):
    """Save model checkpoint."""
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    print(f"✓ Checkpoint saved: {filename}")


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    """Load model checkpoint."""
    print(f"Loading checkpoint: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    # Update learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("✓ Checkpoint loaded")


def train_fn(disc_S, disc_W, gen_W, gen_S, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    """
    Single training epoch.
    
    Args:
        disc_S: Discriminator for Summer images
        disc_W: Discriminator for Winter images
        gen_W: Generator Summer -> Winter
        gen_S: Generator Winter -> Summer
        loader: DataLoader
        opt_disc: Discriminator optimizer
        opt_gen: Generator optimizer
        l1: L1 loss (for cycle consistency)
        mse: MSE loss (for adversarial loss)
        d_scaler: GradScaler for discriminator (mixed precision)
        g_scaler: GradScaler for generator (mixed precision)
    """
    loop = tqdm(loader, leave=True)
    
    for idx, (summer, winter) in enumerate(loop):
        summer = summer.to(config.DEVICE)
        winter = winter.to(config.DEVICE)
        
        # ============================================
        # Train Discriminators
        # ============================================
        with torch.cuda.amp.autocast():
            # Generate fake images
            fake_winter = gen_W(summer)
            fake_summer = gen_S(winter)
            
            # Discriminator Summer
            D_S_real = disc_S(summer)
            D_S_fake = disc_S(fake_summer.detach())
            D_S_real_loss = mse(D_S_real, torch.ones_like(D_S_real))
            D_S_fake_loss = mse(D_S_fake, torch.zeros_like(D_S_fake))
            D_S_loss = D_S_real_loss + D_S_fake_loss
            
            # Discriminator Winter
            D_W_real = disc_W(winter)
            D_W_fake = disc_W(fake_winter.detach())
            D_W_real_loss = mse(D_W_real, torch.ones_like(D_W_real))
            D_W_fake_loss = mse(D_W_fake, torch.zeros_like(D_W_fake))
            D_W_loss = D_W_real_loss + D_W_fake_loss
            
            # Total discriminator loss
            D_loss = (D_S_loss + D_W_loss) / 2
        
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()
        
        # ============================================
        # Train Generators
        # ============================================
        with torch.cuda.amp.autocast():
            # Adversarial loss
            D_S_fake = disc_S(fake_summer)
            D_W_fake = disc_W(fake_winter)
            loss_G_S = mse(D_S_fake, torch.ones_like(D_S_fake))
            loss_G_W = mse(D_W_fake, torch.ones_like(D_W_fake))
            
            # Cycle consistency loss
            cycle_winter = gen_W(fake_summer)
            cycle_summer = gen_S(fake_winter)
            cycle_winter_loss = l1(winter, cycle_winter)
            cycle_summer_loss = l1(summer, cycle_summer)
            
            # Identity loss (optional)
            identity_loss = 0
            if config.LAMBDA_IDENTITY > 0:
                identity_winter = gen_W(winter)
                identity_summer = gen_S(summer)
                identity_winter_loss = l1(winter, identity_winter)
                identity_summer_loss = l1(summer, identity_summer)
                identity_loss = (identity_winter_loss + identity_summer_loss) * config.LAMBDA_IDENTITY
            
            # Total generator loss
            G_loss = (
                loss_G_S + loss_G_W
                + (cycle_winter_loss + cycle_summer_loss) * config.LAMBDA_CYCLE
                + identity_loss
            )
        
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        
        # Update progress bar
        loop.set_postfix(D_loss=D_loss.item(), G_loss=G_loss.item())


def main():
    """Main training function."""
    # Initialize models
    disc_S = Discriminator(in_channels=3).to(config.DEVICE)
    disc_W = Discriminator(in_channels=3).to(config.DEVICE)
    gen_W = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_S = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    
    # Optimizers
    opt_disc = optim.Adam(
        list(disc_S.parameters()) + list(disc_W.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    opt_gen = optim.Adam(
        list(gen_W.parameters()) + list(gen_S.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    
    # Loss functions
    L1 = nn.L1Loss()
    mse = nn.MSELoss()
    
    # Load checkpoints if resuming
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN_S, gen_S, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_W, gen_W, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC_S, disc_S, opt_disc, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC_W, disc_W, opt_disc, config.LEARNING_RATE)
    
    # Dataset and DataLoader
    dataset = CycleGANDataset(
        root_summer=config.TRAIN_DIR + "/trainA",
        root_winter=config.TRAIN_DIR + "/trainB",
        transform=config.transforms,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    
    # Mixed precision scalers
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    
    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{config.NUM_EPOCHS}]")
        train_fn(disc_S, disc_W, gen_W, gen_S, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)
        
        # Save checkpoints every 5 epochs
        if config.SAVE_MODEL and (epoch + 1) % 5 == 0:
            save_checkpoint(gen_S, opt_gen, config.CHECKPOINT_GEN_S)
            save_checkpoint(gen_W, opt_gen, config.CHECKPOINT_GEN_W)
            save_checkpoint(disc_S, opt_disc, config.CHECKPOINT_DISC_S)
            save_checkpoint(disc_W, opt_disc, config.CHECKPOINT_DISC_W)


if __name__ == "__main__":
    config.print_config()
    main()

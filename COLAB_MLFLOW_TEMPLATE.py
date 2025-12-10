"""
Colab Training Notebook with MLflow Integration

This is a reference template showing how to add MLflow tracking to your training loop.
Copy these snippets into your 01_colab_train.ipynb notebook.
"""

# ============================================================================
# SETUP CELL (Run after mounting Drive and installing dependencies)
# ============================================================================

import mlflow
import mlflow.pytorch
import os
from pathlib import Path

# Configure MLflow to save to Drive (persistent storage)
mlflow_dir = '/content/drive/MyDrive/SeasonsGAN/mlruns'
os.makedirs(mlflow_dir, exist_ok=True)

mlflow.set_tracking_uri(f'file://{mlflow_dir}')
mlflow.set_experiment("SeasonsGAN_Training")

print(f"✅ MLflow configured. Tracking URI: {mlflow.get_tracking_uri()}")
print(f"✅ Experiment: {mlflow.get_experiment_by_name('SeasonsGAN_Training').name}")

# ============================================================================
# BEFORE TRAINING LOOP (Log hyperparameters)
# ============================================================================

# Start MLflow run
run = mlflow.start_run(run_name="cyclegan_epoch_200_lr_0.0002")

# Log all hyperparameters
mlflow.log_param("num_epochs", 200)
mlflow.log_param("batch_size", 1)
mlflow.log_param("learning_rate", 0.0002)
mlflow.log_param("lambda_cycle", 10.0)
mlflow.log_param("lambda_identity", 5.0)
mlflow.log_param("num_residual_blocks", 9)
mlflow.log_param("model_architecture", "CycleGAN")
mlflow.log_param("dataset", "Yosemite_Summer2Winter")

print(f"✅ Started MLflow run: {run.info.run_id}")

# ============================================================================
# INSIDE TRAINING LOOP (Log metrics every N steps)
# ============================================================================

# Example: In your main training loop, add this code where you calculate losses
for epoch in range(num_epochs):
    for batch_idx, (real_A, real_B) in enumerate(train_loader):
        
        # ... your training code ...
        # ... calculate losses: loss_D_A, loss_D_B, loss_G, loss_cycle ...
        
        # Log metrics every 50 batches
        global_step = epoch * len(train_loader) + batch_idx
        
        if batch_idx % 50 == 0:
            # Log loss metrics
            mlflow.log_metric("batch_loss_D_A", float(loss_D_A), step=global_step)
            mlflow.log_metric("batch_loss_D_B", float(loss_D_B), step=global_step)
            mlflow.log_metric("batch_loss_G_A", float(loss_G_A), step=global_step)
            mlflow.log_metric("batch_loss_G_B", float(loss_G_B), step=global_step)
            mlflow.log_metric("batch_loss_cycle", float(loss_cycle), step=global_step)
        
        # Log sample images every 200 batches
        if batch_idx % 200 == 0:
            # Save generated images
            os.makedirs('temp_samples', exist_ok=True)
            
            # Log fake_B sample
            sample_path = f'temp_samples/fake_B_e{epoch}_b{batch_idx}.jpg'
            save_image(fake_B[0:1], sample_path)  # Save first image of batch
            mlflow.log_artifact(sample_path, artifact_path=f"samples/epoch_{epoch}")
            
            # Log fake_A sample
            sample_path = f'temp_samples/fake_A_e{epoch}_b{batch_idx}.jpg'
            save_image(fake_A[0:1], sample_path)
            mlflow.log_artifact(sample_path, artifact_path=f"samples/epoch_{epoch}")
    
    # Log epoch-level metrics
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss_D_A: {loss_D_A:.4f}, "
          f"Loss_D_B: {loss_D_B:.4f}, Loss_G: {loss_G_A+loss_G_B:.4f}")
    
    mlflow.log_metric("epoch_loss_D_A", float(loss_D_A), step=epoch)
    mlflow.log_metric("epoch_loss_D_B", float(loss_D_B), step=epoch)
    mlflow.log_metric("epoch_loss_G", float(loss_G_A + loss_G_B), step=epoch)
    
    # Save checkpoints every 5 epochs
    if (epoch + 1) % 5 == 0:
        checkpoint_path = f'{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pt'
        torch.save({
            'epoch': epoch,
            'G_A': gen_A.state_dict(),
            'G_B': gen_B.state_dict(),
            'D_A': disc_A.state_dict(),
            'D_B': disc_B.state_dict(),
        }, checkpoint_path)
        
        mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
        print(f"✅ Saved checkpoint to {checkpoint_path}")

# ============================================================================
# AFTER TRAINING (Log final models and end run)
# ============================================================================

# Save final model weights
torch.save(gen_B.state_dict(), f'{checkpoint_dir}/gen_winter.pth')
torch.save(gen_A.state_dict(), f'{checkpoint_dir}/gen_summer.pth')
torch.save(disc_A.state_dict(), f'{checkpoint_dir}/disc_summer.pth')
torch.save(disc_B.state_dict(), f'{checkpoint_dir}/disc_winter.pth')

# Log final artifacts
mlflow.log_artifact(f'{checkpoint_dir}/gen_winter.pth', artifact_path="final_models")
mlflow.log_artifact(f'{checkpoint_dir}/gen_summer.pth', artifact_path="final_models")

# Log training summary
summary = {
    "total_epochs_completed": num_epochs,
    "final_loss_D_A": float(loss_D_A),
    "final_loss_D_B": float(loss_D_B),
    "final_loss_G": float(loss_G_A + loss_G_B),
}
mlflow.log_dict(summary, "training_summary.json")

# End the run
mlflow.end_run()

print("✅ Training complete! MLflow run ended.")
print(f"📊 View results: mlflow ui --backend-store-uri 'file://{mlflow_dir}'")

# ============================================================================
# BONUS: View MLflow UI
# ============================================================================

# Uncomment and run in Colab to view MLflow dashboard:
# import subprocess
# subprocess.run(['mlflow', 'ui', '--backend-store-uri', f'file://{mlflow_dir}', '--host', '0.0.0.0', '--port', '5000'])
# Then tunnel port 5000 via Colab's ngrok or other method

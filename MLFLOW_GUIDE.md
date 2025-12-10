"""
MLflow Integration Guide for SeasonsGAN Training on Colab

Add this to your Colab notebook to track metrics, parameters, and artifacts.
"""

# ============================================================================
# ADD THIS AT THE START OF YOUR TRAINING CELL (before the loop)
# ============================================================================

import mlflow
import mlflow.pytorch

# Set experiment
mlflow.set_experiment("SeasonsGAN_CycleGAN")

# Start a run
mlflow.start_run(run_name="cycle_gan_training")

# Log parameters
mlflow.log_param("num_epochs", NUM_EPOCHS)
mlflow.log_param("learning_rate", LEARNING_RATE)
mlflow.log_param("batch_size", BATCH_SIZE)
mlflow.log_param("lambda_cycle", LAMBDA_CYCLE)
mlflow.log_param("lambda_identity", LAMBDA_IDENTITY)
mlflow.log_param("num_residual_blocks", NUM_RESIDUAL_BLOCKS)

print("✅ MLflow experiment started: SeasonsGAN_CycleGAN")

# ============================================================================
# ADD THIS INSIDE YOUR TRAINING LOOP (where you calculate losses)
# ============================================================================

# Inside the epoch loop:
for epoch in range(NUM_EPOCHS):
    for idx, (real_a, real_b) in enumerate(train_loader):
        
        # ... your training code ...
        
        # Log metrics every N steps
        step = epoch * len(train_loader) + idx
        
        if idx % 100 == 0:
            mlflow.log_metric("loss_D_A", loss_D_A.item(), step=step)
            mlflow.log_metric("loss_D_B", loss_D_B.item(), step=step)
            mlflow.log_metric("loss_G_A", loss_G_A.item(), step=step)
            mlflow.log_metric("loss_G_B", loss_G_B.item(), step=step)
            mlflow.log_metric("loss_cycle", loss_cycle.item(), step=step)
            mlflow.log_metric("loss_identity", loss_identity.item(), step=step)
        
        # Log sample images every 500 steps
        if idx % 500 == 0:
            # Save generated images
            os.makedirs("temp_images", exist_ok=True)
            img_path = f"temp_images/winter_epoch{epoch}_step{idx}.png"
            save_image(fake_b, img_path)
            mlflow.log_artifact(img_path, artifact_path="generated_images")
    
    # Log epoch summary
    print(f"Epoch {epoch}/{NUM_EPOCHS} - D_A: {loss_D_A:.4f}, D_B: {loss_D_B:.4f}, "
          f"G_A: {loss_G_A:.4f}, G_B: {loss_G_B:.4f}")

# ============================================================================
# ADD THIS AT THE END (after training completes)
# ============================================================================

# Log final models
mlflow.pytorch.log_model(gen_a2b, "generator_a2b")
mlflow.pytorch.log_model(gen_b2a, "generator_b2a")
mlflow.pytorch.log_model(disc_a, "discriminator_a")
mlflow.pytorch.log_model(disc_b, "discriminator_b")

# End the run
mlflow.end_run()

print("✅ MLflow run completed and logged!")

# ============================================================================
# OPTIONAL: View MLflow UI locally
# ============================================================================
# After training, run in terminal:
# mlflow ui --backend-store-uri /content/mlruns
# Then open http://localhost:5000 in your browser

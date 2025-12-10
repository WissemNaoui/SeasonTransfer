# VS Code + Google Colab Workflow Guide

## Setup Overview

You're using **VS Code as the frontend** with **Google Colab as the GPU backend**. This gives you the best of both worlds: VS Code's superior editing experience + Colab's free GPU.

## Initial Setup (One-Time)

### 1. Install VS Code Extension

Install the **"Colab"** extension in VS Code:
- Extension ID: `ms-toolsai.vscode-jupyter-colab`
- Or search "Colab" in VS Code Extensions

### 2. Prepare Local Data

```bash
cd /home/wissem/.gemini/antigravity/scratch/SeasonsGAN
./prepare_data.sh
```

This organizes your `/home/wissem/Downloads/archive` data into the correct structure.

### 3. Upload to Google Drive

Upload these folders to your Google Drive:

```
/MyDrive/SeasonsGAN/
├── data/
│   └── summer2winter_yosemite/
│       ├── trainA/  (summer images)
│       └── trainB/  (winter images)
└── src/
    ├── data/
    ├── models/
    ├── utils/
    ├── training.py
    └── inference.py
```

**Critical:** The `checkpoints/` folder will be created automatically on Drive during training.

## Running Training

### Option A: VS Code + Colab Extension

1. Open `notebooks/01_colab_train.ipynb` in VS Code
2. Click "Select Kernel" → "Colab"
3. Sign in to Google account
4. Select GPU runtime
5. Run cells sequentially

### Option B: Direct Colab (Browser)

1. Upload `notebooks/01_colab_train.ipynb` to Google Colab
2. Runtime → Change runtime type → GPU
3. Run all cells

## Critical: Data Persistence Strategy

### The Problem
Colab runtimes are **ephemeral**. If you disconnect:
- All files in `/content/` are **deleted**
- Training progress is **lost**
- You start from scratch

### The Solution
The notebook **automatically saves everything to Google Drive**:

```python
# Checkpoints saved to Drive every 5 epochs
CHECKPOINT_PATH = '/content/drive/MyDrive/SeasonsGAN/checkpoints'
```

### What This Means
- ✅ Disconnect/reconnect freely
- ✅ Resume training from last checkpoint
- ✅ Access trained models from any device
- ✅ No data loss

## Training Timeline (T4 GPU)

| Epochs | Time | Checkpoint Size |
|--------|------|-----------------|
| 5 | ~30 min | ~200 MB |
| 50 | ~5 hours | ~200 MB |
| 200 | ~20 hours | ~200 MB |

**Strategy:** Train in 50-epoch chunks, verify results, continue if needed.

## Monitoring Training

### In Notebook
- **Progress bar**: Shows current batch/epoch
- **Loss values**: 
  - `D_loss`: Discriminator loss (should stabilize ~0.5)
  - `G_loss`: Generator loss (should decrease then stabilize)

### Expected Loss Behavior
```
Epoch 1:   D_loss ~1.5, G_loss ~5.0  (models learning)
Epoch 50:  D_loss ~0.5, G_loss ~2.0  (converging)
Epoch 200: D_loss ~0.5, G_loss ~1.5  (stable)
```

## Testing Results

After training, test the generator:

```python
from inference import load_generator, transform_image
import torch

device = torch.device("cuda")
gen = load_generator("/content/drive/MyDrive/SeasonsGAN/checkpoints/gen_winter.pth.tar", device)

# Transform summer → winter
output = transform_image("test_summer.jpg", gen, device)
output.save("test_winter.jpg")
```

## Troubleshooting

### "Runtime disconnected"
- **Solution:** Checkpoints are on Drive. Re-run notebook, set `LOAD_MODEL = True` in config.

### "CUDA out of memory"
- **Solution:** Reduce `BATCH_SIZE` (already set to 1, so this is rare).

### "Data not found"
- **Solution:** Verify Drive paths in cell 1 of notebook.

### "Training too slow"
- **Check GPU:** Runtime → Change runtime type → Verify GPU is enabled
- **Check GPU type:** `!nvidia-smi` (T4 is good, K80 is slow, A100 is excellent)

## File Locations Reference

| What | Local | Google Drive |
|------|-------|--------------|
| Project code | `/home/wissem/.gemini/antigravity/scratch/SeasonsGAN/` | `/MyDrive/SeasonsGAN/` |
| Training data | `data/summer2winter_yosemite/` | `/MyDrive/SeasonsGAN/data/` |
| Checkpoints | N/A | `/MyDrive/SeasonsGAN/checkpoints/` |
| Notebook | `notebooks/01_colab_train.ipynb` | Upload to Colab |

## Next Steps

1. ✅ **Run `prepare_data.sh`** to organize your local data
2. ✅ **Upload to Drive** (data + src folders)
3. ✅ **Open notebook in VS Code** with Colab extension
4. ✅ **Start training** and monitor progress
5. ✅ **Test inference** after 50+ epochs

---

**Pro Tip:** Keep the Colab tab open in your browser while training. Colab is more likely to maintain the connection if the tab is active.

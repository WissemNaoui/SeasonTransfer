# Pre-Flight Checklist: Ready to Train?

Use this checklist before starting training to avoid common issues.

## ✅ Google Drive Setup

- [ ] **Folder structure created:**
  ```
  /MyDrive/SeasonsGAN/
  ├── data/
  │   └── summer2winter_yosemite.zip  ← MUST be .zip file
  ├── src/
  │   ├── data/
  │   ├── models/
  │   ├── utils/
  │   ├── training.py
  │   └── inference.py
  └── checkpoints/  ← Will be created automatically
  ```

- [ ] **Zip file uploaded** (not the unzipped folder)
  - File: `summer2winter_yosemite.zip`
  - Size: ~500MB - 2GB (depending on dataset)
  - Location: `/MyDrive/SeasonsGAN/data/`

- [ ] **Source code uploaded**
  - Folder: `src/` with all Python files
  - Location: `/MyDrive/SeasonsGAN/src/`

## ✅ Colab Notebook Setup

- [ ] **Notebook uploaded to Colab**
  - File: `01_colab_train.ipynb`
  - Can be in Drive or uploaded directly to Colab

- [ ] **Runtime set to GPU**
  - Runtime → Change runtime type → Hardware accelerator: **GPU**
  - Verify: Run `!nvidia-smi` (should show GPU info)

- [ ] **Drive mounted successfully**
  - Cell 1 runs without errors
  - You see: "Mounted at /content/drive"

## ✅ Data Verification

- [ ] **Zip file found on Drive**
  ```python
  import os
  zip_path = '/content/drive/MyDrive/SeasonsGAN/data/summer2winter_yosemite.zip'
  print(f"Zip exists: {os.path.exists(zip_path)}")
  print(f"Zip size: {os.path.getsize(zip_path) / 1e6:.1f} MB")
  ```
  Expected: `True` and size > 100 MB

- [ ] **Data copied to local disk**
  - Cell 2 runs without errors
  - You see: "✅ Done! Data is ready on fast local disk."

- [ ] **Images verified**
  ```python
  !ls /content/data/summer2winter_yosemite/trainA | head -5
  !ls /content/data/summer2winter_yosemite/trainB | head -5
  ```
  Expected: List of .jpg files

- [ ] **Image count is reasonable**
  ```python
  !find /content/data/summer2winter_yosemite/trainA -type f | wc -l
  !find /content/data/summer2winter_yosemite/trainB -type f | wc -l
  ```
  Expected: 100+ images in each folder

## ✅ Code Verification

- [ ] **Source code copied to runtime**
  - Cell 3 runs without errors
  - You see: "✓ Project code loaded"

- [ ] **Imports work**
  ```python
  from models.generator import Generator
  from models.discriminator import Discriminator
  from data.dataset import CycleGANDataset
  print("✓ All imports successful")
  ```

- [ ] **Model tests pass**
  - Cell 7 (Test Model Architectures) runs without errors
  - You see: "✓ Generator test passed" and "✓ Discriminator test passed"

## ✅ Training Configuration

- [ ] **Config paths are correct**
  ```python
  from utils import config
  print(f"TRAIN_DIR: {config.TRAIN_DIR}")
  print(f"CHECKPOINT_PATH: {config.CHECKPOINT_GEN_W}")
  ```
  Expected:
  - `TRAIN_DIR`: `/content/data/summer2winter_yosemite`
  - Checkpoints: `/content/drive/.../checkpoints/...`

- [ ] **Hyperparameters reviewed**
  ```python
  config.print_config()
  ```
  Verify:
  - Batch Size: 1 ✓
  - Learning Rate: 2e-4 ✓
  - Epochs: 200 (or your desired number)
  - Lambda Cycle: 10 ✓

## ✅ GPU Verification

- [ ] **GPU is available**
  ```python
  import torch
  print(f"CUDA available: {torch.cuda.is_available()}")
  print(f"GPU name: {torch.cuda.get_device_name(0)}")
  ```
  Expected: `True` and GPU name (T4, A100, etc.)

- [ ] **GPU memory is sufficient**
  ```python
  !nvidia-smi --query-gpu=memory.total,memory.free --format=csv
  ```
  Expected: At least 8GB total (T4 has 16GB)

## ✅ Final Checks

- [ ] **Disk space available**
  ```python
  !df -h /content
  ```
  Expected: At least 5GB free

- [ ] **No previous training artifacts**
  ```python
  # If resuming training, this is OK
  # If starting fresh, checkpoints should not exist yet
  import os
  ckpt_path = '/content/drive/MyDrive/SeasonsGAN/checkpoints/'
  if os.path.exists(ckpt_path):
      print(f"Existing checkpoints: {os.listdir(ckpt_path)}")
  else:
      print("No existing checkpoints (fresh start)")
  ```

## 🚀 Ready to Train!

If all boxes are checked, you're ready to start training:

```python
from training import main
main()
```

## 🔧 Common Issues

### Issue: "Zip file not found"
**Fix:** Verify zip path in Cell 2 matches your Drive structure

### Issue: "CUDA out of memory"
**Fix:** Restart runtime, reduce batch size (already at 1, so this is rare)

### Issue: "No module named 'models'"
**Fix:** Re-run Cell 3 (copy src/ and add to Python path)

### Issue: "Training very slow (< 5 images/sec)"
**Fix:** Verify data is on local disk, not Drive:
```python
from utils import config
print(config.TRAIN_DIR)  # Should be /content/data/..., NOT /content/drive/...
```

---

**Pro Tip:** Run through this checklist BEFORE starting a long training run. It takes 2 minutes and saves hours of debugging.

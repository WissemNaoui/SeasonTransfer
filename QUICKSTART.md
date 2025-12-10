# 🚀 QUICK START: 5 Minutes to Training

## Prerequisites ✅

You said you've already uploaded data to Google Drive. Perfect!

## Step-by-Step (Do This Now)

### 1. Verify Your Drive Has the Zip File (30 seconds)

Open Google Drive in your browser and confirm you have:

```
/MyDrive/SeasonsGAN/data/summer2winter_yosemite.zip
```

**CRITICAL:** It must be a `.zip` file, not an unzipped folder!

If you uploaded the folder instead:
1. Download it back to your local machine
2. Zip it: `cd ~/Downloads && zip -r summer2winter_yosemite.zip summer2winter_yosemite/`
3. Upload the `.zip` to Drive

---

### 2. Upload Source Code to Drive (1 minute)

Upload the entire `src/` folder from your local project to:

```
/MyDrive/SeasonsGAN/src/
```

You can do this via:
- **Option A:** Drag and drop in Google Drive web interface
- **Option B:** Use Google Drive desktop app
- **Option C:** Use `rclone` or similar CLI tool

---

### 3. Open Notebook in Colab (30 seconds)

1. Go to [Google Colab](https://colab.research.google.com)
2. Click **File → Upload notebook**
3. Upload: `/home/wissem/.gemini/antigravity/scratch/SeasonsGAN/notebooks/01_colab_train.ipynb`

---

### 4. Set Runtime to GPU (15 seconds)

In Colab:
1. Click **Runtime → Change runtime type**
2. Hardware accelerator: **GPU**
3. Click **Save**

---

### 5. Run the Notebook (2 minutes setup + training)

Run cells **sequentially** (Shift+Enter for each cell):

#### Cell 1: Mount Drive
```python
# You'll be prompted to authorize
# Click the link, sign in, copy the code
```
**Expected output:** "Mounted at /content/drive"

#### Cell 2: Copy Data to Local Disk ⚡ (NEW - CRITICAL!)
```python
# This copies zip from Drive → unzips to local disk
# Takes ~2 minutes, but makes training 10x faster!
```
**Expected output:** "✅ Done! Data is ready on fast local disk."

#### Cell 3: Install Dependencies
```python
# Installs albumentations, tqdm
```
**Expected output:** "✓ Dependencies installed"

#### Cell 4: Copy Source Code
```python
# Copies src/ from Drive to runtime
```
**Expected output:** "✓ Project code loaded"

#### Cell 5: Verify Data
```python
# Checks that images are accessible
```
**Expected output:** "✓ Found XXX summer images" and "✓ Found XXX winter images"

#### Cell 6: Update Config
```python
# Sets paths for training
```
**Expected output:** Config summary with paths

#### Cell 7: Test Models
```python
# Verifies Generator and Discriminator work
```
**Expected output:** "✓ Generator test passed" and "✓ Discriminator test passed"

#### Cell 8: START TRAINING 🚀
```python
# This is the big one - runs for hours
```
**Expected output:** Progress bars showing epoch/batch progress

---

## 🎯 What to Watch For

### During Training

**Good signs ✅:**
- Progress bar shows ~15-20 images/sec
- `D_loss` starts around 1.5, stabilizes around 0.5
- `G_loss` starts around 5.0, decreases over time
- Checkpoints saved every 5 epochs

**Bad signs ❌:**
- Progress bar shows < 5 images/sec → Data is on Drive, not local disk
- `D_loss` or `G_loss` is NaN → Learning rate too high (rare with default config)
- "CUDA out of memory" → Restart runtime, try again

---

## ⏱️ Timeline

| Checkpoint | Time | What to Do |
|------------|------|------------|
| **5 epochs** | 30 min | Verify training is working, losses are decreasing |
| **50 epochs** | 5 hours | **STOP HERE** - Test inference, check image quality |
| **100 epochs** | 10 hours | Continue if 50-epoch results look promising |
| **200 epochs** | 20 hours | Full training (diminishing returns after this) |

**Recommendation:** Train to 50 epochs first, then decide if you need more.

---

## 🧪 Testing Results (After 50+ Epochs)

After training, run **Cell 9** (Test Inference):

```python
# Loads a summer image, transforms to winter
# Displays side-by-side comparison
```

**What to look for:**
- ✅ Snow added to landscape
- ✅ Colors shifted (green → white/blue)
- ✅ Overall structure preserved
- ❌ Artifacts, distortions (train longer if this happens)

---

## 🔧 Troubleshooting

### "Zip file not found"

**Fix:** Check the path in Cell 2:
```python
drive_zip_path = '/content/drive/MyDrive/SeasonsGAN/data/summer2winter_yosemite.zip'
```
Adjust if your Drive folder name is different.

### "Training very slow (< 5 images/sec)"

**Fix:** Verify data is on local disk:
```python
from utils import config
print(config.TRAIN_DIR)
# Should be: /content/data/summer2winter_yosemite
# NOT: /content/drive/MyDrive/...
```

### "Runtime disconnected"

**Fix:** This is normal for long training runs. When you reconnect:
1. Re-run Cell 1 (Mount Drive)
2. Re-run Cell 2 (Copy data to local disk)
3. Re-run Cell 3-6 (Setup)
4. In Cell 6, set `config.LOAD_MODEL = True`
5. Re-run Cell 8 (Training resumes from last checkpoint)

---

## 📊 Expected Loss Behavior

```
Epoch 1:   D_loss ~1.5, G_loss ~5.0   (models learning from scratch)
Epoch 10:  D_loss ~0.8, G_loss ~3.0   (starting to converge)
Epoch 50:  D_loss ~0.5, G_loss ~2.0   (good results)
Epoch 200: D_loss ~0.5, G_loss ~1.5   (best results)
```

If losses diverge or go to NaN, something is wrong (very rare with default config).

---

## 🎉 Success Criteria

You'll know training is working when:

1. ✅ Progress bar shows 15-20 images/sec
2. ✅ Losses are decreasing (or stable after convergence)
3. ✅ Checkpoints are being saved to Drive every 5 epochs
4. ✅ After 50 epochs, inference produces recognizable winter scenes

---

## 📖 Read These If You Get Stuck

- **CHECKLIST.md** - Pre-flight verification checklist
- **DATA_STRATEGY.md** - Why local disk is 10x faster
- **WORKFLOW.md** - Detailed VS Code + Colab setup

---

## 🚀 Ready? Let's Go!

1. ✅ Verify zip on Drive
2. ✅ Upload src/ to Drive
3. ✅ Open notebook in Colab
4. ✅ Set runtime to GPU
5. ✅ Run cells 1-8

**You're 5 minutes away from training!** 🎨

---

**Questions?** Check the troubleshooting section above or the detailed guides in the docs.

# SeasonsGAN: Summer ↔ Winter Image Translation

CycleGAN implementation for unpaired image-to-image translation between summer and winter landscapes.

## Architecture

- **Generator**: ResNet-based with 9 residual blocks (256x256 images)
- **Discriminator**: PatchGAN with 70x70 receptive field
- **Loss**: Adversarial loss + Cycle consistency loss (λ=10)

## Project Structure

```
SeasonsGAN/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py       # CycleGAN dataset with unpaired loading
│   ├── models/
│   │   ├── __init__.py
│   │   ├── generator.py     # ResNet Generator
│   │   └── discriminator.py # PatchGAN Discriminator
│   ├── utils/
│   │   └── config.py        # Hyperparameters & transforms
│   ├── training.py          # Training loop
│   └── inference.py         # Image transformation
├── notebooks/
│   └── 01_colab_train.ipynb # Colab training notebook
├── checkpoints/             # Saved model weights (on Drive)
├── requirements.txt
└── README.md
```

## Setup for Google Colab

### 1. Upload Data to Google Drive

Upload the `summer2winter_yosemite` dataset to:
```
/MyDrive/SeasonsGAN/data/summer2winter_yosemite/
├── trainA/  (summer images)
└── trainB/  (winter images)
```

### 2. Upload Project Code

Upload the entire `src/` folder to:
```
/MyDrive/SeasonsGAN/src/
```

### 3. Open Notebook in Colab

1. Upload `notebooks/01_colab_train.ipynb` to Colab
2. Set runtime to **GPU** (Runtime → Change runtime type → GPU)
3. Run all cells

### 4. Monitor Training

- Checkpoints saved every 5 epochs to `/MyDrive/SeasonsGAN/checkpoints/`
- Training progress shown via tqdm progress bars
- Loss values: `D_loss` (discriminator), `G_loss` (generator)

## Local Testing (Optional)

```bash
# Install dependencies
pip install -r requirements.txt

# Test generator architecture
python src/models/generator.py

# Test discriminator architecture
python src/models/discriminator.py

# Test dataset loading
python -c "from src.data.dataset import CycleGANDataset; print('✓ Dataset OK')"
```

## Inference

After training, transform images using:

```python
from inference import load_generator, transform_image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = load_generator("checkpoints/gen_winter.pth.tar", device)
output = transform_image("test_summer.jpg", gen, device)
output.save("test_winter.jpg")
```

## Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch Size | 1 | Required for Instance Normalization |
| Learning Rate | 2e-4 | Adam with β1=0.5 |
| Epochs | 200 | ~8-10 hours on T4 GPU |
| λ Cycle | 10 | Cycle consistency weight |
| λ Identity | 0 | Disabled by default |

## Critical: Data Persistence

**Always save checkpoints to Google Drive**, not the Colab runtime!

The notebook automatically:
- Mounts Drive at `/content/drive`
- Saves checkpoints to `/MyDrive/SeasonsGAN/checkpoints/`
- Survives runtime disconnections

## References

- [CycleGAN Paper](https://arxiv.org/abs/1703.10593)
- [Official PyTorch Implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

## License

MIT

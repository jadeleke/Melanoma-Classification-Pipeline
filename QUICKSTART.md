# Quick Start Guide: WOA-Hybrid Model

## Prerequisites

- Python 3.10+
- NVIDIA GPU with 8GB+ VRAM (RTX 5060 or equivalent)
- CUDA 13.0+ toolkit
- 32GB RAM recommended (16GB minimum)

## Installation

1. **Clone and navigate to project**:
```bash
cd segmentation
```

2. **Create virtual environment**:
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Windows CMD
.\.venv\Scripts\activate.bat
# Linux/Mac
source .venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Data Preparation

1. **Download ISIC-2020 dataset** from Kaggle:
   - https://www.kaggle.com/competitions/siim-isic-melanoma-classification
   - Download `jpeg/` folder
   - Extract to `data/train/images/` and `data/test/images/`

2. **Prepare train/val splits**:
```bash
python prepare_data.py
```

Expected output:
- `data/train/train_split.csv`
- `data/train/val_split.csv`

## Training

### Option 1: Hybrid Model (Recommended)

Train with HRNet + EfficientNet + WOA optimization:

```bash
python train_hybrid.py
```

**Training phases**:
- **Warmup** (Epochs 1-10): Standard training, build representations
- **WOA optimization** (Epochs 10-30): Optimize fusion weights every 5 epochs

**Expected training time**: 6-10 hours on RTX 5060

**Outputs**:
- Best model: `checkpoints/best_model.pth`
- Training log: `logs/training_log_YYYYMMDD_HHMMSS.json`
- WOA results: `logs/woa_optimization_YYYYMMDD_HHMMSS.json`

### Option 2: Baseline Model (Faster)

Train EfficientNet-B3 only for comparison:

```bash
python train_enhanced.py
```

**Expected training time**: 3-4 hours

## Configuration

Edit `src/config.py` to customize:

### Quick toggles:
```python
CONFIG = {
    'model_type': 'hybrid',     # 'hybrid' or 'baseline'
    'batch_size': 8,            # Reduce if OOM errors
    'epochs': 30,
    
    # WOA settings
    'woa': {
        'enabled': True,
        'n_whales': 20,         # Reduce for faster optimization
        'max_iter': 30,
    },
    
    # StyleGAN (optional)
    'stylegan': {
        'enabled': False,       # Enable if you have pretrained generator
        'synthetic_ratio': 0.2,
    },
}
```

### Memory optimization:
If you encounter OOM (Out of Memory) errors:

1. **Reduce batch size**:
```python
'batch_size': 6,  # or 4
```

2. **Freeze backbones** after warmup:
```python
'hrnet': {
    'freeze_backbone': True,
},
'efficientnet': {
    'freeze_backbone': True,
},
```

3. **Disable mixed precision** (slower but uses less memory):
```python
'mixed_precision': False,
```

## Monitoring Training

### Real-time progress:
- Training loss and learning rate shown in progress bar
- Validation metrics printed after each epoch
- Fusion weights displayed during validation

### Logs:
```bash
# View training log
cat logs/training_log_*.json

# View WOA optimization results
cat logs/woa_optimization_*.json
```

### TensorBoard (optional):
```bash
tensorboard --logdir=logs/
```

## Evaluation

The training script automatically evaluates on validation set. For custom evaluation:

```python
from src.models.hybrid_model import create_hybrid_model
from src.config import CONFIG
import torch

# Load model
model = create_hybrid_model(CONFIG)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Get feature importance
importance = model.get_feature_importance()
print("Component importance:")
print(f"  HRNet: {importance['hrnet']:.4f}")
print(f"  EfficientNet: {importance['efficientnet']:.4f}")
```

## Expected Performance

### Baseline (EfficientNet-B3):
- Accuracy: 91.99%
- Sensitivity: 90.95%
- Specificity: 92.13%
- AUC-ROC: ~0.95

### Hybrid Model (Target):
- Accuracy: >92%
- Sensitivity: >91%
- Specificity: >92.5%
- AUC-ROC: >0.96

## Troubleshooting

### 1. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size in `config.py`:
```python
'batch_size': 6,  # or 4
```

### 2. HRNet model not found
```
RuntimeError: Unknown model hrnet_w32
```
**Solution**: HRNet will fall back to ResNet50. To use actual HRNet:
```bash
pip install timm --upgrade
```

### 3. StyleGAN import error
```
ModuleNotFoundError: No module named 'stylegan2_pytorch'
```
**Solution**: StyleGAN is optional. Either:
- Disable in config: `'stylegan': {'enabled': False}`
- Install: `pip install stylegan2-pytorch`

### 4. Slow WOA optimization
```
WOA taking too long (>30 minutes)
```
**Solution**: Reduce WOA parameters:
```python
'woa': {
    'n_whales': 10,    # Fewer whales
    'max_iter': 20,    # Fewer iterations
}
```

### 5. Training crashes after warmup
```
Error during WOA optimization
```
**Solution**: Disable WOA temporarily:
```python
'woa': {'enabled': False}
```

## Next Steps

1. **Compare models**:
   - Train both baseline and hybrid
   - Compare metrics and fusion weights
   
2. **Ablation studies**:
   - Train with HRNet frozen
   - Train with EfficientNet frozen
   - Compare different fusion strategies

3. **Hyperparameter tuning**:
   - Try different WOA parameters
   - Experiment with fusion strategies
   - Test different learning rates

4. **Advanced features**:
   - Train/fine-tune StyleGAN generator
   - Implement test-time augmentation
   - Add attention mechanisms

## Resources

- **Architecture details**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **Implementation guide**: [IMPLEMENTATION.md](IMPLEMENTATION.md)
- **Full README**: [README.md](README.md)
- **Configuration reference**: [src/config.py](src/config.py)

## Support

For issues or questions:
1. Check [ARCHITECTURE.md](ARCHITECTURE.md) for detailed explanations
2. Review configuration in `src/config.py`
3. Check logs in `logs/` directory
4. Review error messages in console output

---

**Good luck with training!** 🚀

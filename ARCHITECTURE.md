# WOA-Hybrid Model Architecture

## Overview

This document describes the Whale Optimization Algorithm (WOA) based hybrid deep learning model for skin lesion classification. The model combines multiple state-of-the-art architectures optimized through metaheuristic feature fusion.

## Architecture Components

### 1. HRNet (High-Resolution Network)
- **Purpose**: Extract high-resolution spatial features
- **Model**: HRNet-W32 (can be upgraded to W48)
- **Input**: RGB images (256×256×3)
- **Output**: 2048-dimensional feature vector
- **Key Features**:
  - Maintains high-resolution representations throughout
  - Multi-scale parallel convolutions
  - Excellent for capturing fine-grained spatial details
  - Pretrained on ImageNet

### 2. EfficientNet-B3
- **Purpose**: Extract deep classification features
- **Model**: EfficientNet-B3
- **Input**: RGB images (256×256×3)
- **Output**: 1536-dimensional feature vector
- **Key Features**:
  - Compound scaling (depth, width, resolution)
  - Efficient architecture with MBConv blocks
  - Pretrained on ImageNet
  - Strong transfer learning capabilities

### 3. StyleGAN2 (Optional)
- **Purpose**: Data augmentation through synthetic image generation
- **Usage**: Training phase only
- **Key Features**:
  - Generate synthetic lesion images
  - Style mixing for diversity
  - Controllable image quality (truncation)
  - Can be disabled without affecting inference

### 4. WOA (Whale Optimization Algorithm)
- **Purpose**: Optimize feature fusion weights
- **Optimization Target**: Maximize validation AUC-ROC
- **Parameters**:
  - Population size: 20-30 whales
  - Iterations: 30-50
  - Search space: [0, 1] (normalized weights)
- **Key Features**:
  - Nature-inspired metaheuristic
  - Balances exploration and exploitation
  - Finds optimal weight combination
  - Runs periodically during training

## Model Pipeline

```
Input Image (256×256×3)
    ↓
┌───────────────────────────────┐
│   Parallel Feature Extraction │
├───────────────┬───────────────┤
│    HRNet      │  EfficientNet │
│  (Spatial)    │ (Classification)│
│      ↓        │       ↓        │
│   Features    │   Features     │
│   (2048-D)    │   (1536-D)     │
└───────┬───────┴───────┬───────┘
        │               │
        └───────┬───────┘
                ↓
    ┌───────────────────────┐
    │  WOA-Optimized Fusion │
    │  w₁ × F_HRNet +       │
    │  w₂ × F_EfficientNet  │
    └───────────┬───────────┘
                ↓
    ┌───────────────────────┐
    │   Classification Head │
    │   • Linear(fused, 512)│
    │   • BatchNorm + ReLU  │
    │   • Dropout(0.5)      │
    │   • Linear(512, 2)    │
    └───────────┬───────────┘
                ↓
        Output Logits (2 classes)
        [Benign, Malignant]
```

## Feature Fusion Strategies

### Weighted Sum (Default)
```python
fused_features = w₁ × HRNet_features + w₂ × EfficientNet_features
```
- Weights optimized by WOA
- Normalized: w₁ + w₂ = 1
- Allows dynamic balance between components
- More interpretable

### Concatenation (Alternative)
```python
fused_features = concat(HRNet_features, EfficientNet_features)
```
- Fixed combination
- Total dimension: 2048 + 1536 = 3584
- No weight optimization needed
- Higher dimensional space

## Training Strategy

### Phase 1: Warmup (Epochs 1-10)
1. Train all components with standard optimization
2. Fusion weights learn through gradients
3. Build initial feature representations
4. No WOA optimization yet

**Hyperparameters**:
- Learning rate: 1e-4 (backbone), 1e-4 (fusion)
- Batch size: 8 (effective 32 with accumulation)
- Mixed precision: FP16
- StyleGAN ratio: 0% initially

### Phase 2: WOA Optimization (Epochs 10-30)
1. Continue standard training
2. Run WOA every 5 epochs:
   - Evaluate 20-30 weight combinations
   - Maximize validation AUC
   - Update fusion weights
3. Fine-tune with optimized weights

**WOA Schedule**:
- Epochs 10, 15, 20, 25, 30: Run WOA optimization
- Each run: 30 iterations, 20 whales
- Convergence criterion: ΔScore < 1e-4

### Component-wise Learning Rates
```python
HRNet backbone:      1e-5  (0.1× base)
EfficientNet:        1e-5  (0.1× base)
Fusion layers:       1e-4  (1.0× base)
Classifier:          1e-4  (1.0× base)
```

## Loss Function

**Focal Loss** (default for imbalanced data):
```
FL(pₜ) = -αₜ(1 - pₜ)ᵞ log(pₜ)
```
- α = 0.25 (adjustable based on class distribution)
- γ = 2.0 (focus on hard examples)
- Label smoothing: 0.1

**Alternative**: Weighted Cross-Entropy

## Memory Optimization

**GPU Memory Budget (RTX 5060 8GB)**:
- Model parameters: ~2.5 GB
- Activations (batch=8): ~1.5 GB
- Gradients: ~2.5 GB
- Optimizer states: ~1.0 GB
- Buffer: ~0.5 GB
- **Total**: ~8 GB

**Optimization Techniques**:
1. **Mixed Precision (FP16)**: 40% memory reduction
2. **Gradient Accumulation**: Effective batch size without memory cost
3. **Gradient Checkpointing**: Can be enabled if OOM
4. **Component Freezing**: Freeze backbones after warmup (optional)

## Performance Metrics

### Classification Metrics
- **Accuracy**: Overall correctness
- **Sensitivity**: True positive rate (critical for medical)
- **Specificity**: True negative rate
- **AUC-ROC**: Area under ROC curve (optimization target)
- **AUC-PR**: Area under precision-recall curve

### Hybrid Model Metrics
- **Fusion Weights**: Component importance
- **WOA Convergence**: Optimization quality
- **Component Ablation**: Individual contributions

## Configuration Options

### Model Selection
```python
CONFIG = {
    'model_type': 'hybrid',  # or 'baseline' for EfficientNet-only
    
    'hrnet': {
        'model_name': 'hrnet_w32',      # or 'hrnet_w48'
        'feature_dim': 2048,
        'freeze_backbone': False,       # True for faster training
    },
    
    'efficientnet': {
        'model_name': 'efficientnet_b3',
        'feature_dim': 1536,
        'freeze_backbone': False,
    },
    
    'fusion': {
        'strategy': 'weighted_sum',     # or 'concat'
        'initial_weights': [0.5, 0.5],
        'learnable': True,              # Gradient-based learning
    },
    
    'woa': {
        'enabled': True,
        'n_whales': 20,
        'max_iter': 30,
        'optimization_frequency': 5,    # Every N epochs
        'start_epoch': 10,              # After warmup
    },
    
    'stylegan': {
        'enabled': False,               # Optional augmentation
        'synthetic_ratio': 0.2,         # 20% synthetic images
    },
}
```

## Ablation Study Results

| Configuration | Accuracy | Sensitivity | Specificity | AUC-ROC |
|--------------|----------|-------------|-------------|---------|
| EfficientNet-only (baseline) | 91.99% | 90.95% | 92.13% | ~0.95 |
| HRNet-only | TBD | TBD | TBD | TBD |
| HRNet + EfficientNet (equal weights) | TBD | TBD | TBD | TBD |
| HRNet + EfficientNet (WOA) | **Target** | **Target** | **Target** | **>0.96** |
| + StyleGAN augmentation | TBD | TBD | TBD | TBD |

## Inference Pipeline

### 1. Single Image Inference
```python
model.eval()
with torch.no_grad():
    # Preprocess image
    image = preprocess(image)  # Resize + Normalize
    
    # Extract features
    hrnet_features = model.hrnet(image)
    eff_features = model.efficientnet(image)
    
    # Fuse features
    fused = model.fusion(hrnet_features, eff_features)
    
    # Classify
    logits = model.classifier(fused)
    probs = softmax(logits)
    
    prediction = argmax(probs)
```

### 2. Batch Inference
- Batch size: 16-32 (larger than training)
- No augmentation
- Test-time augmentation (TTA) optional

### 3. Model Export
- **PyTorch (.pth)**: Full model + optimizer state
- **ONNX (.onnx)**: For deployment
- **TorchScript (.pt)**: Optimized inference

## File Structure

```
src/
├── models/
│   ├── __init__.py
│   ├── hrnet.py              # HRNet feature extractor
│   ├── efficientnet.py       # EfficientNet feature extractor
│   ├── stylegan.py           # StyleGAN augmentation
│   ├── woa_optimizer.py      # WOA algorithm
│   └── hybrid_model.py       # Combined architecture
├── config.py                 # Configuration
├── dataset.py                # Data loading
├── loss.py                   # Loss functions
├── metrics.py                # Evaluation metrics
└── train.py                  # Training utilities

train_hybrid.py               # Main training script
checkpoints/                  # Model checkpoints
logs/                         # Training logs + WOA results
results/                      # Evaluation results
```

## Comparison with Literature

### State-of-the-Art Models on ISIC-2020
| Model | Approach | AUC-ROC | Year |
|-------|----------|---------|------|
| HSSA-based | Hybrid metaheuristic | ~0.96 | 2023 |
| IWO-based | Invasive weed optimization | ~0.95 | 2023 |
| EfficientNet-B7 | Transfer learning | ~0.94 | 2021 |
| **WOA-Hybrid (This)** | **WOA + Multi-architecture** | **Target >0.96** | **2024** |

## Advantages

1. **Multi-scale Features**: HRNet captures fine details, EfficientNet captures semantic information
2. **Optimized Fusion**: WOA finds optimal balance automatically
3. **Interpretability**: Fusion weights show component importance
4. **Flexibility**: Components can be frozen/unfrozen dynamically
5. **Augmentation**: Optional StyleGAN for data diversity
6. **Memory Efficient**: Runs on consumer GPU (8GB)

## Limitations & Future Work

1. **Training Time**: ~6-10 hours (vs 3-4 hours for baseline)
2. **Model Size**: ~150M parameters (vs 12M for baseline)
3. **WOA Overhead**: 10-15 minutes per optimization
4. **StyleGAN Dependency**: Optional but requires additional training

**Future Improvements**:
- Knowledge distillation for smaller model
- Neural architecture search (NAS) for optimal structure
- Multi-dataset training for better generalization
- Attention mechanisms for feature fusion

## References

1. HRNet: Deep High-Resolution Representation Learning (CVPR 2019)
2. EfficientNet: Rethinking Model Scaling (ICML 2019)
3. StyleGAN2: Analyzing and Improving Image Synthesis (CVPR 2020)
4. Whale Optimization Algorithm (Advances in Engineering Software 2016)
5. ISIC 2020 Challenge: https://challenge2020.isic-archive.com/

---

**Last Updated**: November 2024
**Authors**: Research Team
**License**: MIT

"""
Configuration file for skin cancer detection training.
Based on TODO.md specifications for RTX 5060 8GB VRAM.
"""

import torch
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"

# Dataset paths (update these after downloading data)
TRAIN_IMG_DIR = DATA_DIR / "train" / "images"
TRAIN_CSV = DATA_DIR / "train" / "labels.csv"
TEST_IMG_DIR = DATA_DIR / "test" / "images"
METADATA_CSV = DATA_DIR / "metadata.csv"

# Hyperparameters (optimized for RTX 5060 8GB VRAM)
CONFIG = {
    # Data
    'image_size': 256,              # Input resolution (optimized for 8GB VRAM)
    'batch_size': 8,                # Adjusted for RTX 5060 8GB VRAM
    'num_workers': 4,               # DataLoader workers
    
    # Model (backward compatibility)
    'model_name': 'efficientnet_b3', # Target model
    'pretrained': True,
    'num_classes': 2,               # Malignant vs Benign
    'dropout': 0.3,
    
    # Model type: 'baseline' or 'hybrid'
    'model_type': 'hybrid',         # Use 'baseline' for original EfficientNet-only
    
    # HRNet configuration
    'hrnet': {
        'model_name': 'hrnet_w32',  # or 'hrnet_w48' for better quality
        'pretrained': True,
        'feature_dim': 2048,
        'freeze_backbone': False,   # Set True to freeze HRNet backbone
    },
    
    # EfficientNet configuration (for hybrid model)
    'efficientnet': {
        'model_name': 'efficientnet_b3',
        'pretrained': True,
        'feature_dim': 1536,        # EfficientNet-B3 default output
        'dropout': 0.3,
        'freeze_backbone': False,   # Set True to freeze EfficientNet backbone
    },
    
    # StyleGAN configuration
    'stylegan': {
        'enabled': False,           # Enable StyleGAN augmentation
        'generator_path': None,     # Path to pretrained generator (optional)
        'synthetic_ratio': 0.2,     # Ratio of synthetic images (0-1)
        'latent_dim': 512,
        'truncation_psi': 0.7,      # Lower = higher quality, less diversity
        'use_style_mixing': True,   # Enable style mixing for diversity
        'mixing_prob': 0.5,
    },
    
    # Feature fusion configuration
    'fusion': {
        'strategy': 'weighted_sum', # 'weighted_sum' or 'concat'
        'initial_weights': [0.5, 0.5],  # [HRNet, EfficientNet]
        'learnable': True,          # Allow gradient-based learning
        'hidden_dim': 512,          # Classifier hidden dimension
        'dropout': 0.5,
    },
    
    # WOA (Whale Optimization Algorithm) configuration
    'woa': {
        'enabled': True,            # Enable WOA optimization
        'n_whales': 20,             # Population size (20-30 recommended)
        'max_iter': 30,             # Max iterations (30-50 recommended)
        'optimization_frequency': 5, # Optimize every N epochs
        'start_epoch': 10,          # Start WOA after warmup
        'verbose': True,
    },
    
    # Training
    'epochs': 30,                   # Single training run (no experiments)
    'learning_rate': 1e-4,
    'weight_decay': 1e-2,
    'warmup_epochs': 5,
    
    # Loss
    'focal_alpha': 0.25,            # Will be adjusted based on class distribution
    'focal_gamma': 2.0,
    'label_smoothing': 0.1,
    
    # Optimization
    'gradient_accumulation': 4,     # Effective batch = 8 × 4 = 32
    'gradient_clip': 1.0,
    # Enable mixed precision only when CUDA is available
    'mixed_precision': bool(torch.cuda.is_available()),
    
    # Regularization
    'early_stopping_patience': 10,  # Reduced for shorter training
    'reduce_lr_patience': 5,
    'reduce_lr_factor': 0.5,
    
    # Data split
    'train_split': 0.8,
    'val_split': 0.2,
    'random_seed': 42,
    
    # ImageNet normalization
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# Augmentation parameters
AUGMENTATION_CONFIG = {
    'horizontal_flip_p': 0.5,
    'vertical_flip_p': 0.5,
    'rotate90_p': 0.5,
    'shift_scale_rotate': {
        'shift_limit': 0.1,
        'scale_limit': 0.1,
        'rotate_limit': 45,
        'p': 0.5
    },
    'brightness_contrast': {
        'brightness_limit': 0.2,
        'contrast_limit': 0.2,
        'p': 0.5
    },
    'hue_saturation_value': {
        'hue_shift_limit': 20,
        'sat_shift_limit': 30,
        'val_shift_limit': 20,
        'p': 0.3
    },
    'blur_p': 0.3,
    'coarse_dropout_p': 0.3,
    'random_gamma_p': 0.3,
}

def print_config():
    """Print configuration for verification."""
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Device: {CONFIG['device']}")
    if CONFIG['device'] == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"\nImage Size: {CONFIG['image_size']}x{CONFIG['image_size']}")
    print(f"Batch Size: {CONFIG['batch_size']} (effective: {CONFIG['batch_size'] * CONFIG['gradient_accumulation']})")
    print(f"Model: {CONFIG['model_name']}")
    print(f"Epochs: {CONFIG['epochs']}")
    print(f"Learning Rate: {CONFIG['learning_rate']}")
    print(f"Mixed Precision: {CONFIG['mixed_precision']}")
    print("="*60 + "\n")

if __name__ == "__main__":
    print_config()

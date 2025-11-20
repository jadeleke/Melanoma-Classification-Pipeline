"""
Custom PyTorch Dataset for SIIM-ISIC Melanoma Classification.
Handles image loading, caching, and augmentations.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut


def load_dicom_image(dicom_path):
    """
    Load and process DICOM image to RGB format.
    
    Args:
        dicom_path: Path to .dcm file
    
    Returns:
        image: RGB numpy array (H, W, 3)
    """
    # Read DICOM file
    dicom = pydicom.dcmread(str(dicom_path))
    
    # Apply VOI LUT (Value of Interest Lookup Table) for proper windowing
    image = apply_voi_lut(dicom.pixel_array, dicom)
    
    # Normalize to 0-255 range (safe against zero max)
    image = image - image.min()
    max_val = float(image.max())
    if max_val == 0 or np.isnan(max_val) or np.isinf(max_val):
        # Avoid division by zero; leave image as zeros
        image = np.zeros_like(image, dtype=np.uint8)
    else:
        image = image / max_val
        image = (image * 255).astype(np.uint8)
    
    # Convert grayscale to RGB (repeat channels)
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=2)
    
    return image


class MelanomaDataset(Dataset):
    """
    SIIM-ISIC Melanoma Classification Dataset.
    Supports both JPEG (.jpg) and DICOM (.dcm) formats.
    Extended to support StyleGAN augmentation.
    
    Args:
        csv_path: Path to CSV with image_name and target columns
        img_dir: Directory containing images (JPEG or DICOM)
        image_size: Target image size (default: 256)
        transform: Albumentations transform pipeline
        mode: 'train' or 'val' for different augmentation strategies
        image_format: 'jpg' or 'dcm' (default: auto-detect)
        stylegan_augmenter: StyleGAN augmenter for synthetic images (optional)
        synthetic_ratio: Ratio of synthetic images to add (0-1)
    """
    
    def __init__(self, csv_path, img_dir, image_size=256, transform=None, mode='train', 
                 image_format='auto', stylegan_augmenter=None, synthetic_ratio=0.0):
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.image_size = image_size
        self.transform = transform
        self.mode = mode
        self.stylegan_augmenter = stylegan_augmenter
        self.synthetic_ratio = synthetic_ratio if mode == 'train' else 0.0
        
        # Auto-detect format from first file
        if image_format == 'auto':
            first_img_name = self.df.iloc[0]['image_name']
            if (self.img_dir / f"{first_img_name}.dcm").exists():
                self.image_format = 'dcm'
            elif (self.img_dir / f"{first_img_name}.jpg").exists():
                self.image_format = 'jpg'
            else:
                # Try to find any file with this name
                possible_files = list(self.img_dir.glob(f"{first_img_name}.*"))
                if possible_files:
                    self.image_format = possible_files[0].suffix[1:]  # Remove the dot
                else:
                    raise FileNotFoundError(f"Could not find image: {first_img_name}")
        else:
            self.image_format = image_format
        
        print(f"📁 Dataset initialized with {self.image_format.upper()} format")
        
        # Cache for loaded images (optional, uses more RAM but faster)
        self.cache = {}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image_name']
        target = row['target']
        
        # Decide if using synthetic image
        use_synthetic = (self.mode == 'train' and 
                        self.stylegan_augmenter is not None and 
                        self.synthetic_ratio > 0 and 
                        np.random.random() < self.synthetic_ratio)
        
        if use_synthetic:
            # Generate synthetic image using StyleGAN
            synthetic_tensor = self.stylegan_augmenter.generate_batch(1, self.image_size)
            # Convert from [-1, 1] to [0, 1] and then to [0, 255]
            image = ((synthetic_tensor[0].cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
            image = np.transpose(image, (1, 2, 0))  # CHW to HWC
            img_name = f"synthetic_{idx}"
        else:
            # Load real image (check cache first)
            if img_name in self.cache:
                image = self.cache[img_name]
            else:
                img_path = self.img_dir / f"{img_name}.{self.image_format}"
                
                # Load based on format
                if self.image_format == 'dcm':
                    image = load_dicom_image(img_path)
                else:  # jpg, jpeg, png, etc.
                    image = cv2.imread(str(img_path))
                    if image is None:
                        raise FileNotFoundError(f"Could not load image: {img_path}")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Optionally cache if dataset is small enough
                # self.cache[img_name] = image
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return {
            'image': image,
            'target': torch.tensor(target, dtype=torch.long),
            'image_name': img_name,
            'is_synthetic': use_synthetic
        }


def get_train_transforms(image_size=256, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Training augmentation pipeline following TODO.md specifications.
    Heavy augmentation for skin lesion images.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        
        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=45,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
        
        # Color augmentations
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.3
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        
        # Noise and blur
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.CoarseDropout(
            max_holes=8,
            max_height=image_size // 8,
            max_width=image_size // 8,
            fill_value=0,
            p=0.3
        ),
        
        # Normalization
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_val_transforms(image_size=256, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Validation/test augmentation pipeline.
    Only resize and normalize, no data augmentation.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def create_dataloaders(train_csv, val_csv, img_dir, config, stylegan_augmenter=None):
    """
    Create train and validation DataLoaders.
    
    Args:
        train_csv: Path to training CSV
        val_csv: Path to validation CSV
        img_dir: Directory containing images
        config: Configuration dictionary
        stylegan_augmenter: StyleGAN augmenter for synthetic images (optional)
    
    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader
    
    # Get StyleGAN config
    stylegan_config = config.get('stylegan', {})
    synthetic_ratio = stylegan_config.get('synthetic_ratio', 0.0) if stylegan_config.get('enabled', False) else 0.0
    
    # Create datasets
    train_dataset = MelanomaDataset(
        csv_path=train_csv,
        img_dir=img_dir,
        image_size=config['image_size'],
        transform=get_train_transforms(
            image_size=config['image_size'],
            mean=config['mean'],
            std=config['std']
        ),
        mode='train',
        stylegan_augmenter=stylegan_augmenter if synthetic_ratio > 0 else None,
        synthetic_ratio=synthetic_ratio
    )
    
    val_dataset = MelanomaDataset(
        csv_path=val_csv,
        img_dir=img_dir,
        image_size=config['image_size'],
        transform=get_val_transforms(
            image_size=config['image_size'],
            mean=config['mean'],
            std=config['std']
        ),
        mode='val',
        stylegan_augmenter=None,
        synthetic_ratio=0.0
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True if config['device'] == 'cuda' else False,
        drop_last=True  # For stable batch norm
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'] * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset implementation...")
    
    # This will fail until data is downloaded, but shows the interface
    try:
        from config import CONFIG
        
        test_dataset = MelanomaDataset(
            csv_path="dummy.csv",
            img_dir="dummy_dir",
            image_size=CONFIG['image_size'],
            transform=get_train_transforms(),
            mode='train'
        )
        print("✓ Dataset class created successfully")
    except Exception as e:
        print(f"Expected error (no data yet): {e}")
    
    print("\nAugmentation pipeline:")
    print("Train transforms:", get_train_transforms())
    print("\nVal transforms:", get_val_transforms())

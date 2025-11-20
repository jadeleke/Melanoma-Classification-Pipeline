"""
Data preparation script for SIIM-ISIC Melanoma Classification.
Downloads and organizes the dataset for training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil


def prepare_siim_isic_data(data_dir, train_split=0.8, random_seed=42):
    """
    Prepare SIIM-ISIC dataset for training.
    
    Expected structure after download:
    data/
        train/
            images/  (JPEG images)
            train.csv  (image_name, target columns)
        test/
            images/  (JPEG images)
    
    This function will:
    1. Load training CSV
    2. Create stratified train/val split
    3. Save train.csv and val.csv
    4. Print dataset statistics
    
    Args:
        data_dir: Path to data directory
        train_split: Fraction for training (default: 0.8)
        random_seed: Random seed for reproducibility
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    train_csv_path = train_dir / "train.csv"
    
    print("="*60)
    print("SIIM-ISIC DATA PREPARATION")
    print("="*60)
    
    # Check if data exists
    if not train_csv_path.exists():
        print(f"\n❌ Error: {train_csv_path} not found!")
        print("\nPlease download the SIIM-ISIC dataset from Kaggle:")
        print("https://www.kaggle.com/competitions/siim-isic-melanoma-classification/data")
        print("\nExpected structure:")
        print("data/")
        print("  train/")
        print("    images/  (JPEG files)")
        print("    train.csv")
        print("  test/")
        print("    images/  (JPEG files)")
        return False
    
    # Load training data
    print(f"\n📁 Loading data from {train_csv_path}...")
    df = pd.read_csv(train_csv_path)
    
    # Print dataset statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total samples: {len(df):,}")
    print(f"  Benign (0):    {(df['target'] == 0).sum():,} ({(df['target'] == 0).sum() / len(df) * 100:.2f}%)")
    print(f"  Malignant (1): {(df['target'] == 1).sum():,} ({(df['target'] == 1).sum() / len(df) * 100:.2f}%)")
    print(f"  Columns: {list(df.columns)}")
    
    # Check for required columns
    if 'image_name' not in df.columns or 'target' not in df.columns:
        print("\n❌ Error: CSV must contain 'image_name' and 'target' columns")
        return False
    
    # Create stratified split
    print(f"\n🔀 Creating stratified train/val split ({train_split:.0%}/{1-train_split:.0%})...")
    train_df, val_df = train_test_split(
        df,
        train_size=train_split,
        stratify=df['target'],
        random_state=random_seed
    )
    
    # Save splits
    train_output = train_dir / "train_split.csv"
    val_output = train_dir / "val_split.csv"
    
    train_df.to_csv(train_output, index=False)
    val_df.to_csv(val_output, index=False)
    
    print(f"\n✅ Split created successfully!")
    print(f"  Training set:   {len(train_df):,} samples -> {train_output}")
    print(f"    Benign:       {(train_df['target'] == 0).sum():,}")
    print(f"    Malignant:    {(train_df['target'] == 1).sum():,}")
    print(f"  Validation set: {len(val_df):,} samples -> {val_output}")
    print(f"    Benign:       {(val_df['target'] == 0).sum():,}")
    print(f"    Malignant:    {(val_df['target'] == 1).sum():,}")
    
    # Calculate class weights for loss function
    class_counts = [
        (train_df['target'] == 0).sum(),
        (train_df['target'] == 1).sum()
    ]
    print(f"\n⚖️  Class distribution for Focal Loss:")
    print(f"  Class 0 (Benign):   {class_counts[0]:,}")
    print(f"  Class 1 (Malignant): {class_counts[1]:,}")
    print(f"  Recommended alpha:  {class_counts[0] / sum(class_counts):.4f}")
    
    print("\n" + "="*60)
    print("✅ Data preparation complete!")
    print("="*60 + "\n")
    
    return True


def download_instructions():
    """Print download instructions."""
    print("\n" + "="*60)
    print("DATASET DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("\n1. Go to: https://www.kaggle.com/competitions/siim-isic-melanoma-classification/data")
    print("\n2. Accept competition rules and download:")
    print("   - train.zip")
    print("   - test.zip")
    print("\n3. Extract to your data directory:")
    print("   data/")
    print("     train/")
    print("       images/ (33,126 JPEG files)")
    print("       train.csv")
    print("     test/")
    print("       images/ (10,982 JPEG files)")
    print("\n4. Run this script again: python prepare_data.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Path to data directory
    from src.config import DATA_DIR
    
    # Prepare data
    success = prepare_siim_isic_data(
        data_dir=DATA_DIR,
        train_split=0.8,
        random_seed=42
    )
    
    if not success:
        download_instructions()

"""
Complete automated training pipeline for Google Colab.
Downloads data from Kaggle, prepares dataset, and trains model.

Usage in Colab:
    1. Upload this file and src/ folder to Colab
    2. Upload kaggle.json when prompted
    3. Run: python main.py
    4. Wait ~2.5 hours for training

Note: All files (code, data, checkpoints) will be in /content/segmentation/
"""

import os
import sys
import json
import zipfile
import shutil
from pathlib import Path
import subprocess


def check_colab_environment():
    """Check if running in Google Colab and setup working directory."""
    try:
        import google.colab
        in_colab = True
        print("✓ Running in Google Colab")
    except ImportError:
        in_colab = False
        print("⚠️  Not running in Colab (running locally)")
    
    # Setup working directory
    work_dir = Path('/content/segmentation') if in_colab else Path.cwd()
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)
    
    print(f"✓ Working directory: {work_dir}")
    return in_colab, work_dir


def setup_kaggle_credentials():
    """Setup Kaggle API credentials."""
    print("\n" + "="*60)
    print("KAGGLE CREDENTIALS CHECK")
    print("="*60)
    
    kaggle_config = Path.home() / '.kaggle' / 'kaggle.json'
    
    if kaggle_config.exists():
        with open(kaggle_config, 'r') as f:
            creds = json.load(f)
            print(f"\n✓ Kaggle credentials found")
            print(f"  Username: {creds.get('username', 'N/A')}")
        return True
    
    print("\n❌ Kaggle credentials not found!")
    print("\nTo setup Kaggle API:")
    print("  1. Go to https://www.kaggle.com/settings")
    print("  2. Click 'Create New Token'")
    print("  3. Place kaggle.json in ~/.kaggle/")
    print("  4. Run this script again\n")
    
    return False


def check_gpu():
    """Verify GPU is available."""
    print("\n" + "="*60)
    print("GPU VERIFICATION")
    print("="*60)
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("\n⚠️  WARNING: GPU not available!")
            print("Training will be VERY slow on CPU.")
            print("\nTo enable GPU in Colab:")
            print("  Runtime → Change runtime type → GPU (T4)")
            response = input("\nContinue anyway? (y/N): ")
            if response.lower() != 'y':
                sys.exit(1)
            return False
        
        print(f"\n✓ PyTorch: {torch.__version__}")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"✓ CUDA: {torch.version.cuda}")
        
        return True
        
    except ImportError:
        print("\n❌ PyTorch not installed!")
        print("Installing dependencies...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch', 'torchvision'], check=True)
        return check_gpu()


def install_dependencies():
    """Install required Python packages."""
    print("\n" + "="*60)
    print("INSTALLING DEPENDENCIES")
    print("="*60)
    
    packages = [
        'torch',
        'torchvision', 
        'timm',
        'albumentations',
        'opencv-python-headless',
        'scikit-learn',
        'pandas',
        'tqdm',
        'kaggle'
    ]
    
    print("\nInstalling packages (this may take 2-3 minutes)...")
    
    try:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-q'] + packages,
            check=True,
            capture_output=False
        )
        print("\n✓ All dependencies installed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to install dependencies: {e}")
        return False


def download_dataset():
    """Download SIIM-ISIC dataset from Kaggle."""
    print("\n" + "="*60)
    print("DOWNLOADING DATASET FROM KAGGLE")
    print("="*60)
    
    data_dir = Path('data')
    train_dir = data_dir / 'train' / 'images'
    test_dir = data_dir / 'test' / 'images'
    
    # Check if already downloaded
    if train_dir.exists() and len(list(train_dir.glob('*.jpg'))) > 30000:
        print("\n✓ Dataset already downloaded")
        train_count = len(list(train_dir.glob('*.jpg')))
        test_count = len(list(test_dir.glob('*.jpg')))
        print(f"  Training images: {train_count:,}")
        print(f"  Test images: {test_count:,}")
        return True
    
    # Create data directory
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📥 Downloading SIIM-ISIC Melanoma dataset...")
    print("   This will download the ENTIRE competition dataset")
    print("   Size: ~10-15 GB total")
    print("   Time: 15-30 minutes depending on connection")
    print("   Progress may appear stalled - this is normal\n")
    
    try:
        # Download entire competition dataset
        os.chdir('data')
        
        print("📦 Downloading all competition files...")
        print("   (This includes JPEG, DICOM, TFRecords, and metadata)")
        subprocess.run([
            'kaggle', 'competitions', 'download',
            '-c', 'siim-isic-melanoma-classification'
        ], check=True)
        
        os.chdir('..')
        
        # Extract files
        print("\n📂 Extracting files...")
        
        # Find and extract all zip files
        zip_files = list(data_dir.glob('*.zip'))
        print(f"   Found {len(zip_files)} zip file(s) to extract")
        
        for zip_file in zip_files:
            print(f"   Extracting {zip_file.name}...")
            try:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
                zip_file.unlink()  # Delete zip after extraction
            except zipfile.BadZipFile:
                print(f"   ⚠️  Skipping bad zip file: {zip_file.name}")
                continue
        
        # Organize images from jpeg folder
        print("\n📁 Organizing images from jpeg folder...")
        
        jpeg_folder = data_dir / 'jpeg'
        if jpeg_folder.exists():
            # Move train images
            jpeg_train = jpeg_folder / 'train'
            if jpeg_train.exists():
                train_dir.mkdir(parents=True, exist_ok=True)
                train_images = list(jpeg_train.glob('*.jpg'))
                print(f"   Moving {len(train_images)} training images...")
                for img in train_images:
                    shutil.move(str(img), str(train_dir / img.name))
            
            # Move test images
            jpeg_test = jpeg_folder / 'test'
            if jpeg_test.exists():
                test_dir.mkdir(parents=True, exist_ok=True)
                test_images = list(jpeg_test.glob('*.jpg'))
                print(f"   Moving {len(test_images)} test images...")
                for img in test_images:
                    shutil.move(str(img), str(test_dir / img.name))
            
            # Remove empty jpeg folder structure
            print("   Cleaning up empty folders...")
            shutil.rmtree(jpeg_folder)
        else:
            print("   ⚠️  Warning: jpeg folder not found in extracted data")
        
        # Move train.csv if it exists
        train_csv = data_dir / 'train' / 'train.csv'
        if not train_csv.exists():
            # Check if it's in the data root
            root_csv = data_dir / 'train.csv'
            if root_csv.exists():
                (data_dir / 'train').mkdir(parents=True, exist_ok=True)
                shutil.move(str(root_csv), str(train_csv))
        
        # Cleanup other folders if needed
        cleanup_dirs = [data_dir / '__MACOSX', data_dir / 'train' / '__MACOSX']
        for cleanup_dir in cleanup_dirs:
            if cleanup_dir.exists():
                shutil.rmtree(cleanup_dir)
        
        # Verify download
        train_count = len(list(train_dir.glob('*.jpg'))) if train_dir.exists() else 0
        test_count = len(list(test_dir.glob('*.jpg'))) if test_dir.exists() else 0
        
        print("\n✓ Dataset downloaded and extracted!")
        print(f"  Training images: {train_count:,}")
        print(f"  Test images: {test_count:,}")
        
        if train_count < 30000:
            print("\n⚠️  Warning: Expected ~33,000 training images")
            print(f"   Only found {train_count} images")
            return False
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to download dataset: {e}")
        print("\nTroubleshooting:")
        print("  1. Check kaggle.json is valid")
        print("  2. Accept competition rules at:")
        print("     https://www.kaggle.com/competitions/siim-isic-melanoma-classification/rules")
        print("  3. Ensure you have enough disk space (~15 GB)")
        return False
    
    except Exception as e:
        print(f"\n❌ Error during download: {e}")
        import traceback
        traceback.print_exc()
        return False


def prepare_data():
    """Run prepare_data.py to create train/val splits."""
    print("\n" + "="*60)
    print("PREPARING DATASET (TRAIN/VAL SPLIT)")
    print("="*60)
    
    if not Path('prepare_data.py').exists():
        print("\n❌ prepare_data.py not found!")
        print("Please upload prepare_data.py to the current directory.")
        return False
    
    try:
        print("\nCreating stratified train/validation split...\n")
        result = subprocess.run(
            [sys.executable, 'prepare_data.py'],
            check=True,
            capture_output=False
        )
        
        # Verify splits were created
        train_split = Path('data/train/train_split.csv')
        val_split = Path('data/train/val_split.csv')
        
        if train_split.exists() and val_split.exists():
            print("\n✓ Data preparation complete!")
            return True
        else:
            print("\n❌ Split files not created")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Data preparation failed: {e}")
        return False
    
    except FileNotFoundError:
        print("\n❌ Python interpreter not found")
        return False


def train_model():
    """Run train_model.py to train the model."""
    print("\n" + "="*60)
    print("STARTING MODEL TRAINING")
    print("="*60)
    
    if not Path('train_model.py').exists():
        print("\n❌ train_model.py not found!")
        print("Please upload train_model.py to the current directory.")
        return False
    
    if not Path('src').exists():
        print("\n❌ src/ directory not found!")
        print("Please upload the src/ folder with all Python modules.")
        return False
    
    try:
        print("\n🚀 Starting training...")
        print("   Expected time: ~2-2.5 hours on T4 GPU")
        print("   Keep browser tab open to prevent timeout")
        print("   Checkpoints saved every epoch\n")
        print("="*60 + "\n")
        
        result = subprocess.run(
            [sys.executable, 'train_model.py'],
            check=True,
            capture_output=False
        )
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Check for best model
        best_model = Path('checkpoints/best_model.pth')
        if best_model.exists():
            size_mb = best_model.stat().st_size / 1024**2
            print(f"\n✓ Best model saved: {best_model}")
            print(f"  Size: {size_mb:.1f} MB")
        
        # Check for training logs
        log_file = Path('logs/training_history.csv')
        if log_file.exists():
            print(f"\n✓ Training logs: {log_file}")
        
        print("\n💡 Next steps:")
        print("  1. Download trained model from Files panel")
        print("  2. View training curves in logs/")
        print("  3. Use model for inference locally")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        print("\nCheck the error messages above for details.")
        return False
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Latest checkpoint saved in checkpoints/last_model.pth")
        return False
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False


def main():
    """Main pipeline execution."""
    print("\n" + "="*60)
    print("SKIN CANCER DETECTION TRAINING PIPELINE")
    print("SIIM-ISIC Melanoma Classification")
    print("="*60)
    print("\nThis script will:")
    print("  1. Check environment (Colab/Local)")
    print("  2. Verify Kaggle credentials")
    print("  3. Verify GPU availability")
    print("  4. Install dependencies")
    print("  5. Download dataset (~10 GB)")
    print("  6. Prepare train/val splits")
    print("  7. Train model (~2.5 hours)")
    print("\nTotal time: ~3 hours")
    print("="*60)
    
    # Get user confirmation
    response = input("\nReady to start? (Y/n): ")
    if response.lower() == 'n':
        print("Aborted.")
        return
    
    try:
        # Step 0: Check environment
        in_colab, work_dir = check_colab_environment()
        
        # Step 1: Verify Kaggle credentials
        if not setup_kaggle_credentials():
            print("\n❌ Failed to setup Kaggle credentials")
            return
        
        # Step 2: Check GPU
        has_gpu = check_gpu()
        
        # Step 3: Install dependencies
        if not install_dependencies():
            print("\n❌ Failed to install dependencies")
            return
        
        # Step 4: Download dataset
        if not download_dataset():
            print("\n❌ Failed to download dataset")
            return
        
        # Step 5: Prepare data
        if not prepare_data():
            print("\n❌ Failed to prepare data")
            return
        
        # Step 6: Train model
        if not train_model():
            print("\n❌ Training failed or interrupted")
            return
        
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nYour trained model is ready!")
        print("Check the checkpoints/ folder for the best model.")
        print("\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        print("You can resume by running main.py again.")
        print("Already downloaded data will be reused.\n")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

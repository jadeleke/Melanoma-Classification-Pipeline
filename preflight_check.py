"""
Pre-flight check script: Verify all components before training.
Run this before starting training to ensure everything is ready.
"""

import sys
from pathlib import Path

def check_component(name, check_func, critical=True):
    """Check a single component."""
    try:
        result = check_func()
        if result:
            print(f"✅ {name}")
            return True
        else:
            symbol = "❌" if critical else "⚠️"
            print(f"{symbol} {name} - Failed")
            return False
    except Exception as e:
        symbol = "❌" if critical else "⚠️"
        print(f"{symbol} {name} - Error: {e}")
        return False

def main():
    """Run all pre-flight checks."""
    print("\n" + "="*80)
    print("PRE-FLIGHT CHECK: WOA-Hybrid Model Training")
    print("="*80 + "\n")
    
    all_passed = True
    
    # 1. Check Python version
    print("1. Environment Checks")
    print("-" * 40)
    
    def check_python_version():
        version = sys.version_info
        if version.major >= 3 and version.minor >= 10:
            print(f"   Python {version.major}.{version.minor}.{version.micro}")
            return True
        print(f"   Python {version.major}.{version.minor}.{version.micro} (need 3.10+)")
        return False
    
    all_passed &= check_component("Python version (>=3.10)", check_python_version)
    
    # 2. Check required packages
    print("\n2. Package Dependencies")
    print("-" * 40)
    
    required_packages = [
        ('torch', True),
        ('timm', True),
        ('albumentations', True),
        ('cv2', True, 'opencv-python'),
        ('numpy', True),
        ('pandas', True),
        ('sklearn', True, 'scikit-learn'),
        ('tqdm', True),
        ('matplotlib', True),
        ('seaborn', True),
    ]
    
    for pkg_info in required_packages:
        pkg_name = pkg_info[0]
        critical = pkg_info[1]
        display_name = pkg_info[2] if len(pkg_info) > 2 else pkg_name
        
        def check_package():
            __import__(pkg_name)
            return True
        
        all_passed &= check_component(f"{display_name}", check_package, critical)
    
    # 3. Check CUDA availability
    print("\n3. GPU Configuration")
    print("-" * 40)
    
    def check_cuda():
        import torch
        if torch.cuda.is_available():
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        print("   CUDA not available (will use CPU)")
        return True  # Not critical, can train on CPU
    
    check_component("CUDA GPU", check_cuda, critical=False)
    
    # 4. Check source files
    print("\n4. Source Code Components")
    print("-" * 40)
    
    src_files = [
        ('src/config.py', True),
        ('src/dataset.py', True),
        ('src/loss.py', True),
        ('src/metrics.py', True),
        ('src/models/__init__.py', True),
        ('src/models/hrnet.py', True),
        ('src/models/efficientnet.py', True),
        ('src/models/stylegan.py', True),
        ('src/models/woa_optimizer.py', True),
        ('src/models/hybrid_model.py', True),
        ('train_hybrid.py', True),
    ]
    
    for file_path, critical in src_files:
        def check_file():
            return Path(file_path).exists()
        
        all_passed &= check_component(file_path, check_file, critical)
    
    # 5. Check data files
    print("\n5. Data Files")
    print("-" * 40)
    
    data_files = [
        ('data/train/train_split.csv', True),
        ('data/train/val_split.csv', True),
        ('data/train/images', True),
    ]
    
    for file_path, critical in data_files:
        def check_data():
            path = Path(file_path)
            return path.exists()
        
        all_passed &= check_component(file_path, check_data, critical)
    
    # 6. Check model imports
    print("\n6. Model Imports")
    print("-" * 40)
    
    def check_hybrid_model_import():
        from src.models.hybrid_model import create_hybrid_model
        return True
    
    def check_woa_import():
        from src.models.woa_optimizer import WhaleOptimizer
        return True
    
    def check_config_import():
        from src.config import CONFIG
        return True
    
    all_passed &= check_component("Hybrid model import", check_hybrid_model_import)
    all_passed &= check_component("WOA optimizer import", check_woa_import)
    all_passed &= check_component("Config import", check_config_import)
    
    # 7. Check configuration
    print("\n7. Configuration Validation")
    print("-" * 40)
    
    def check_hybrid_config():
        from src.config import CONFIG
        required_keys = ['hrnet', 'efficientnet', 'fusion', 'woa']
        for key in required_keys:
            if key not in CONFIG:
                print(f"   Missing config key: {key}")
                return False
        print(f"   Model type: {CONFIG.get('model_type', 'baseline')}")
        print(f"   Fusion strategy: {CONFIG['fusion']['strategy']}")
        print(f"   WOA enabled: {CONFIG['woa'].get('enabled', False)}")
        return True
    
    all_passed &= check_component("Hybrid model config", check_hybrid_config)
    
    # 8. Check directories
    print("\n8. Output Directories")
    print("-" * 40)
    
    directories = ['checkpoints', 'logs', 'results']
    
    for dir_name in directories:
        def check_dir():
            path = Path(dir_name)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                print(f"   Created: {dir_name}/")
            return True
        
        check_component(dir_name, check_dir, critical=False)
    
    # Summary
    print("\n" + "="*80)
    if all_passed:
        print("✅✅✅ ALL CHECKS PASSED - READY FOR TRAINING ✅✅✅")
        print("\nRun training with:")
        print("    python train_hybrid.py")
    else:
        print("❌ SOME CHECKS FAILED - Please fix issues before training")
        print("\nTo install missing packages:")
        print("    pip install -r requirements.txt")
    print("="*80 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

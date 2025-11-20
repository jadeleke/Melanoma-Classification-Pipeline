"""
Enhanced training script with comprehensive logging, visualization, and testing.
"""

# Suppress noisy albumentations network-version warnings (offline or restricted networks)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"albumentations.check_version")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
import time
from datetime import datetime

from src.config import CONFIG
from src.model import create_model
from src.dataset import MelanomaDataset, create_dataloaders
from src.train import Trainer
from src.tester import ModelTester
from src.visualizer import TrainingVisualizer


def run_training_with_testing():
    """Complete training pipeline with testing and visualization."""
    print(f"\\n{'='*80}")
    print("MELANOMA CLASSIFICATION - COMPLETE TRAINING PIPELINE")
    print(f"{'='*80}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Print configuration
    print(f"\\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Create model
    print(f"\\nCreating model: {CONFIG['model_name']}")
    model = create_model(CONFIG)
    
    # Create data loaders
    print(f"\nLoading training data...")
    
    # Define paths
    from pathlib import Path
    train_csv = Path("data/train/train_split.csv")
    val_csv = Path("data/train/val_split.csv") 
    img_dir = Path("data/train/images")
    
    # Check if data files exist
    if not train_csv.exists():
        print(f"Error: Training data not found at {train_csv}")
        print("Please run: python prepare_data.py")
        return
    
    # Get class counts from training data
    import pandas as pd
    train_df = pd.read_csv(train_csv)
    class_counts = [
        (train_df['target'] == 0).sum(),
        (train_df['target'] == 1).sum()
    ]
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        train_csv=train_csv,
        val_csv=val_csv,
        img_dir=img_dir,
        config=CONFIG
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Class distribution: {class_counts}")
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, CONFIG, class_counts)
    
    # Start training
    print(f"\\n{'='*80}")
    print("STARTING TRAINING")
    print(f"{'='*80}")
    
    start_time = time.time()
    best_metric = trainer.train()
    training_time = time.time() - start_time
    
    print(f"\\n{'='*80}")
    print("TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"Training time: {training_time/3600:.2f} hours")
    print(f"Best validation AUC-ROC: {best_metric:.4f}")
    
    # Load best model for testing
    print(f"\nLoading best model for testing...")
    best_model_path = Path("checkpoints") / "best_model.pth"

    if best_model_path.exists():
        # torch.load behavior changed in PyTorch 2.6: try safe load first, then fall back to full load
        checkpoint = None
        try:
            checkpoint = torch.load(best_model_path, map_location=CONFIG['device'])
            # If this succeeds, it likely returned a state dict or checkpoint
        except Exception as e:
            print(f"Initial torch.load failed: {e}")
            print("Retrying torch.load with weights_only=False (trusted local checkpoint)")
            try:
                checkpoint = torch.load(best_model_path, map_location=CONFIG['device'], weights_only=False)
            except TypeError:
                # Older torch versions may not accept weights_only arg; re-raise original
                raise
            except Exception as e2:
                print(f"Failed to load checkpoint with weights_only=False: {e2}")
                raise

        # Accept either a full checkpoint dict or a state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint.get('epoch', 'unknown')}")
        elif isinstance(checkpoint, dict):
            # assume it's a state_dict
            try:
                model.load_state_dict(checkpoint)
                print("Loaded model state_dict from checkpoint file")
            except Exception as e:
                print(f"Failed to load state_dict from checkpoint dict: {e}")
                raise
        else:
            # If checkpoint is a state_dict-like object
            try:
                model.load_state_dict(checkpoint)
                print("Loaded model state_dict from checkpoint")
            except Exception as e:
                print(f"Unexpected checkpoint format and failed to load: {e}")
                raise
    else:
        print("Warning: Best model checkpoint not found, using current model state")
    
    # Use validation set for comprehensive evaluation (like a test set)
    print(f"\\n{'='*80}")
    print("COMPREHENSIVE VALIDATION SET EVALUATION")
    print(f"{'='*80}")
    
    val_csv = Path("data/train/val_split.csv") 
    val_img_dir = Path("data/train/images")
    
    if val_csv.exists() and val_img_dir.exists():
        print(f"Running comprehensive evaluation on validation set...")
        print(f"Validation CSV: {val_csv}")
        print(f"Images directory: {val_img_dir}")
        
        tester = ModelTester(model, CONFIG, CONFIG['device'])
        val_metrics, predictions_df = tester.run_comprehensive_test(val_csv, val_img_dir)
        
        print(f"\\n{'='*60}")
        print("FINAL VALIDATION RESULTS (COMPREHENSIVE)")
        print(f"{'='*60}")
        print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Validation AUC-ROC: {val_metrics['auc_roc']:.4f}")
        print(f"Validation Sensitivity: {val_metrics['sensitivity']:.4f}")
        print(f"Validation Specificity: {val_metrics['specificity']:.4f}")
        print(f"Validation F1-Score: {val_metrics['f1']:.4f}")
        
    else:
        print(f"\\nSkipping validation evaluation (validation data not found)")
        print(f"Expected: {val_csv} and {val_img_dir}")
    
    # Final summary
    print(f"\\n{'='*80}")
    print("PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {(time.time() - start_time)/3600:.2f} hours")
    print(f"Results saved to:")
    print(f"  - Checkpoints: checkpoints/")
    print(f"  - Logs: logs/")  
    print(f"  - Visualizations: results/")
    if val_csv.exists():
        print(f"  - Validation evaluation: results/testing/")
    print(f"\\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\\n")


def answer_validation_question():
    """Answer the user's question about validation after each epoch."""
    print(f"\\n{'='*60}")
    print("WHY VALIDATION AFTER EACH EPOCH?")
    print(f"{'='*60}")
    print("""
This is standard machine learning practice for several important reasons:

1. **Early Stopping**: Monitor validation performance to stop training when 
   the model starts overfitting (validation performance stops improving).

2. **Model Selection**: Save the best model based on validation performance,
   not the final epoch. The best model often occurs before training ends.

3. **Learning Rate Scheduling**: Adjust learning rate based on validation 
   performance plateaus.

4. **Monitoring**: Track training progress and detect issues like:
   - Overfitting (training improves but validation degrades)
   - Underfitting (both training and validation perform poorly)
   - Learning rate too high/low

5. **Hyperparameter Tuning**: Validation metrics guide hyperparameter 
   adjustments for future runs.

Without per-epoch validation, you would:
- Risk severe overfitting
- Miss the optimal stopping point
- Have no feedback during long training runs
- Be unable to tune hyperparameters effectively

The validation set is separate from the test set:
- Validation: Used during training for model selection
- Test: Used ONLY after training for final evaluation
    """)
    print(f"{'='*60}\\n")


if __name__ == "__main__":
    # Answer the validation question
    answer_validation_question()
    
    # Run complete training pipeline
    try:
        run_training_with_testing()
    except KeyboardInterrupt:
        print("\\nTraining interrupted by user")
    except Exception as e:
        print(f"\\nError during training: {e}")
        import traceback
        traceback.print_exc()
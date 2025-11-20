"""
Training script for WOA-based Hybrid Model.
Combines HRNet + EfficientNet with WOA-optimized feature fusion.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from tqdm import tqdm
import time
import json
from pathlib import Path
from datetime import datetime

from src.config import CONFIG
from src.models.hybrid_model import create_hybrid_model
from src.models.woa_optimizer import FusionWeightOptimizer
from src.models.stylegan import create_stylegan_augmenter
from src.dataset import create_dataloaders
from src.loss import create_loss_fn
from src.metrics import MetricsCalculator


class HybridTrainer:
    """
    Training manager for WOA-based hybrid model.
    
    Features:
    - Dual-phase training: warmup + WOA optimization
    - Mixed precision training (FP16)
    - Gradient accumulation
    - Component-wise learning rate
    - WOA optimization for fusion weights
    - StyleGAN augmentation support
    """
    
    def __init__(self, model, train_loader, val_loader, config, class_counts=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config['device']
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Loss function
        self.criterion = create_loss_fn(config, class_counts)
        
        # Optimizer with component-wise learning rates
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config['epochs'] - config['warmup_epochs'],
            T_mult=1,
            eta_min=1e-6
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config['mixed_precision'] else None
        
        # WOA optimizer for fusion weights
        woa_config = config.get('woa', {})
        if woa_config.get('enabled', True):
            self.woa_optimizer = FusionWeightOptimizer(
                model=self.model,
                val_loader=self.val_loader,
                device=self.device,
                woa_config=woa_config
            )
        else:
            self.woa_optimizer = None
        
        # Metrics tracking
        self.metrics_calculator = MetricsCalculator(track_fusion_weights=True)
        
        # Tracking
        self.current_epoch = 0
        self.best_val_metric = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.train_losses = []
        self.val_losses = []
        self.val_metrics_history = []
        self.fusion_weight_history = []
        
        # Directories
        self.checkpoint_dir = Path("checkpoints")
        self.log_dir = Path("logs")
        self.results_dir = Path("results")
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _create_optimizer(self):
        """Create optimizer with component-wise learning rates."""
        # Different learning rates for different components
        hrnet_lr = self.config['learning_rate'] * 0.1  # Lower LR for pretrained HRNet
        eff_lr = self.config['learning_rate'] * 0.1     # Lower LR for pretrained EfficientNet
        fusion_lr = self.config['learning_rate']        # Full LR for fusion layers
        
        param_groups = [
            {'params': self.model.hrnet.parameters(), 'lr': hrnet_lr, 'name': 'hrnet'},
            {'params': self.model.efficientnet.parameters(), 'lr': eff_lr, 'name': 'efficientnet'},
            {'params': self.model.classifier.parameters(), 'lr': fusion_lr, 'name': 'classifier'},
        ]
        
        # Add fusion weights if learnable
        if hasattr(self.model, 'fusion_weights') and self.model.fusion_weights.requires_grad:
            param_groups.append({
                'params': [self.model.fusion_weights], 
                'lr': fusion_lr, 
                'name': 'fusion_weights'
            })
        
        optimizer = AdamW(
            param_groups,
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        return optimizer
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        num_batches = len(self.train_loader)
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}/{self.config['epochs']}")
        
        # Gradient accumulation
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Forward pass with mixed precision (new torch.amp API)
            if self.config['mixed_precision'] and torch.cuda.is_available():
                from torch.amp import autocast as amp_autocast
                amp_ctx = amp_autocast('cuda')
            else:
                from contextlib import nullcontext
                amp_ctx = nullcontext()

            with amp_ctx:
                outputs = self.model(images)
                outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e6, neginf=-1e6)
                loss = self.criterion(outputs, targets)
                
                # Scale loss for gradient accumulation
                loss = loss / self.config['gradient_accumulation']

            if torch.isnan(loss) or torch.isinf(loss):
                # Skip bad batch
                pbar.set_postfix({'loss': 'nan', 'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"})
                self.optimizer.zero_grad()
                continue
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.config['gradient_accumulation'] == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['gradient_clip']
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['gradient_clip']
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Update running loss
            running_loss += loss.item() * self.config['gradient_accumulation']
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss / (batch_idx + 1):.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # Average loss for epoch
        epoch_loss = running_loss / num_batches
        self.train_losses.append(epoch_loss)
        
        return epoch_loss
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        
        # Reset metrics
        self.metrics_calculator.reset()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                images = batch['image'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e6, neginf=-1e6)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                
                # Calculate predictions
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                # Update metrics
                self.metrics_calculator.update(preds, probs, targets)
        
        # Compute metrics
        val_loss = running_loss / len(self.val_loader)
        metrics = self.metrics_calculator.compute()
        
        self.val_losses.append(val_loss)
        self.val_metrics_history.append(metrics)
        
        # Track fusion weights
        if hasattr(self.model, 'get_feature_importance'):
            fusion_weights = self.model.get_feature_importance()
            self.fusion_weight_history.append(fusion_weights)
            self.metrics_calculator.update_fusion_weights(fusion_weights)
        
        return val_loss, metrics
    
    def optimize_fusion_weights(self):
        """Run WOA optimization for fusion weights."""
        if self.woa_optimizer is None:
            return
        
        woa_config = self.config.get('woa', {})
        
        # Check if we should run WOA this epoch
        if self.current_epoch < woa_config.get('start_epoch', 10):
            return
        
        if (self.current_epoch - woa_config.get('start_epoch', 10)) % woa_config.get('optimization_frequency', 5) != 0:
            return
        
        print(f"\n🐋 Running WOA optimization at epoch {self.current_epoch}...")
        
        # Run WOA optimization
        best_weights, best_score, convergence_curve = self.woa_optimizer.optimize()
        
        print(f"✓ WOA optimization completed")
        print(f"   Best validation AUC: {best_score:.4f}")
        print(f"   Optimized weights: HRNet={best_weights[0]:.4f}, EfficientNet={best_weights[1]:.4f}")
        
        # Save WOA results
        woa_results = {
            'epoch': self.current_epoch,
            'best_weights': best_weights.tolist(),
            'best_score': float(best_score),
            'convergence_curve': [float(x) for x in convergence_curve]
        }
        
        woa_log_path = self.log_dir / f"woa_optimization_{self.timestamp}.json"
        with open(woa_log_path, 'a') as f:
            f.write(json.dumps(woa_results) + '\n')
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_metric': self.best_val_metric,
            'config': self.config,
            'fusion_weights': self.model.fusion_weights.detach().cpu().numpy() if hasattr(self.model, 'fusion_weights') else None,
        }
        
        # Regular checkpoint
        if self.current_epoch % 5 == 0:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pth"
            torch.save(checkpoint, checkpoint_path)
        
        # Best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved (AUC: {self.best_val_metric:.4f})")
    
    def early_stopping(self):
        """Check if training should stop early."""
        patience = self.config.get('early_stopping_patience', 10)
        return self.epochs_without_improvement >= patience
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*80)
        print("STARTING HYBRID MODEL TRAINING")
        print("="*80)
        print(f"Model type: WOA-Hybrid (HRNet + EfficientNet)")
        print(f"Device: {self.device}")
        print(f"Total epochs: {self.config['epochs']}")
        print(f"Batch size: {self.config['batch_size']} (effective: {self.config['batch_size'] * self.config['gradient_accumulation']})")
        print(f"Mixed precision: {self.config['mixed_precision']}")
        print(f"WOA optimization: {'Enabled' if self.woa_optimizer else 'Disabled'}")
        print("="*80 + "\n")
        
        start_time = time.time()
        
        for epoch in range(1, self.config['epochs'] + 1):
            self.current_epoch = epoch
            
            print(f"\nEpoch {epoch}/{self.config['epochs']}")
            print("-" * 50)
            
            # Training
            train_loss = self.train_epoch()
            print(f"Train Loss: {train_loss:.4f}")
            
            # Validation
            val_loss, metrics = self.validate()
            print(f"Val Loss: {val_loss:.4f}")
            self.metrics_calculator.print_metrics(
                fusion_weights=self.model.fusion_weights if hasattr(self.model, 'fusion_weights') else None
            )
            
            # Learning rate scheduling
            if epoch > self.config['warmup_epochs']:
                self.scheduler.step()
            
            # WOA optimization
            self.optimize_fusion_weights()
            
            # Check for improvement
            current_metric = metrics['auc_roc']
            if current_metric > self.best_val_metric:
                self.best_val_metric = current_metric
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                self.save_checkpoint(is_best=True)
            else:
                self.epochs_without_improvement += 1
                print(f"No improvement for {self.epochs_without_improvement} epochs")
            
            # Regular checkpoint
            self.save_checkpoint(is_best=False)
            
            # Early stopping
            if self.early_stopping():
                print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
                break
        
        # Training complete
        elapsed_time = time.time() - start_time
        print("\n" + "="*80)
        print("TRAINING COMPLETED")
        print("="*80)
        print(f"Total time: {elapsed_time / 3600:.2f} hours")
        print(f"Best validation AUC: {self.best_val_metric:.4f} (epoch {self.best_epoch})")
        print("="*80 + "\n")
        
        # Save training log
        self.save_training_log()
        
        # Generate visualizations
        self.generate_visualizations()
    
    def save_training_log(self):
        """Save complete training log."""
        log_data = {
            'timestamp': self.timestamp,
            'config': self.config,
            'total_epochs': self.current_epoch,
            'best_epoch': self.best_epoch,
            'best_val_metric': float(self.best_val_metric),
            'train_losses': [float(x) for x in self.train_losses],
            'val_losses': [float(x) for x in self.val_losses],
            'fusion_weight_history': self.fusion_weight_history,
            'final_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) 
                            for k, v in self.val_metrics_history[-1].items() 
                            if k != 'confusion_matrix'}
        }
        
        log_path = self.log_dir / f"training_log_{self.timestamp}.json"
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"✓ Training log saved to {log_path}")
    
    def generate_visualizations(self):
        """Generate training visualizations."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            print("\nGenerating visualizations...")
            
            # 1. Training curves
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss curves
            epochs = range(1, len(self.train_losses) + 1)
            axes[0, 0].plot(epochs, self.train_losses, label='Train Loss', linewidth=2)
            axes[0, 0].plot(epochs, self.val_losses, label='Val Loss', linewidth=2)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)
            
            # AUC-ROC curve
            if self.val_metrics_history:
                aucs = [m.get('auc_roc', 0) for m in self.val_metrics_history]
                axes[0, 1].plot(epochs, aucs, color='green', linewidth=2, marker='o')
                axes[0, 1].axhline(y=self.best_val_metric, color='r', linestyle='--', label=f'Best: {self.best_val_metric:.4f}')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('AUC-ROC')
                axes[0, 1].set_title('Validation AUC-ROC')
                axes[0, 1].legend()
                axes[0, 1].grid(alpha=0.3)
            
            # Fusion weights evolution
            if self.fusion_weight_history and len(self.fusion_weight_history) > 0:
                hrnet_weights = [w.get('hrnet', 0.5) if isinstance(w, dict) else w[0] for w in self.fusion_weight_history]
                eff_weights = [w.get('efficientnet', 0.5) if isinstance(w, dict) else w[1] for w in self.fusion_weight_history]
                
                weight_epochs = range(1, len(hrnet_weights) + 1)
                axes[1, 0].plot(weight_epochs, hrnet_weights, label='HRNet', linewidth=2, marker='s')
                axes[1, 0].plot(weight_epochs, eff_weights, label='EfficientNet', linewidth=2, marker='o')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Fusion Weight')
                axes[1, 0].set_title('Fusion Weight Evolution (WOA-Optimized)')
                axes[1, 0].legend()
                axes[1, 0].grid(alpha=0.3)
                axes[1, 0].set_ylim([0, 1])
            
            # Final metrics
            if self.val_metrics_history:
                final_metrics = self.val_metrics_history[-1]
                metric_names = ['accuracy', 'sensitivity', 'specificity', 'f1', 'auc_roc']
                metric_values = [final_metrics.get(m, 0) for m in metric_names]
                
                colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(metric_names)))
                axes[1, 1].bar(range(len(metric_names)), metric_values, color=colors, alpha=0.8, edgecolor='black')
                axes[1, 1].set_xticks(range(len(metric_names)))
                axes[1, 1].set_xticklabels([m.replace('_', ' ').title() for m in metric_names], rotation=45, ha='right')
                axes[1, 1].set_ylabel('Score')
                axes[1, 1].set_title('Final Validation Metrics')
                axes[1, 1].set_ylim([0, 1.05])
                axes[1, 1].grid(alpha=0.3, axis='y')
            
            plt.tight_layout()
            viz_path = self.results_dir / f"training_visualization_{self.timestamp}.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Training visualization saved to {viz_path}")
            
            # 2. Confusion matrix
            if self.val_metrics_history:
                final_metrics = self.val_metrics_history[-1]
                if 'confusion_matrix' in final_metrics:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    cm = final_metrics['confusion_matrix']
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                              xticklabels=['Benign', 'Malignant'],
                              yticklabels=['Benign', 'Malignant'],
                              ax=ax)
                    ax.set_ylabel('True Label')
                    ax.set_xlabel('Predicted Label')
                    ax.set_title(f'Final Confusion Matrix (AUC: {self.best_val_metric:.4f})')
                    
                    cm_path = self.results_dir / f"confusion_matrix_{self.timestamp}.png"
                    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"✓ Confusion matrix saved to {cm_path}")
            
        except Exception as e:
            print(f"⚠️  Visualization generation failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main training function."""
    from src.config import CONFIG, CHECKPOINT_DIR, DATA_DIR
    
    # Print configuration
    print("\n" + "="*80)
    print("HYBRID MODEL CONFIGURATION")
    print("="*80)
    print(f"Model type: {CONFIG.get('model_type', 'hybrid')}")
    print(f"HRNet: {CONFIG['hrnet']['model_name']}")
    print(f"EfficientNet: {CONFIG['efficientnet']['model_name']}")
    print(f"Fusion strategy: {CONFIG['fusion']['strategy']}")
    print(f"StyleGAN enabled: {CONFIG['stylegan'].get('enabled', False)}")
    print(f"WOA enabled: {CONFIG['woa'].get('enabled', True)}")
    print("="*80 + "\n")
    
    # Create StyleGAN augmenter if enabled
    stylegan_augmenter = None
    if CONFIG['stylegan'].get('enabled', False):
        stylegan_augmenter = create_stylegan_augmenter(CONFIG)
    
    # Create data loaders
    train_csv = DATA_DIR / "train" / "train_split.csv"
    val_csv = DATA_DIR / "train" / "val_split.csv"
    img_dir = DATA_DIR / "train" / "images"
    
    print("Loading data...")
    train_loader, val_loader = create_dataloaders(
        train_csv=train_csv,
        val_csv=val_csv,
        img_dir=img_dir,
        config=CONFIG,
        stylegan_augmenter=stylegan_augmenter
    )
    
    # Calculate class counts for loss function
    import pandas as pd
    train_df = pd.read_csv(train_csv)
    class_counts = train_df['target'].value_counts().sort_index().values
    print(f"Class distribution: {class_counts}")
    
    # Create hybrid model
    print("Creating hybrid model...")
    model = create_hybrid_model(CONFIG)
    print(f"Total parameters: {model.get_num_parameters():,}")
    
    # Create trainer
    trainer = HybridTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=CONFIG,
        class_counts=class_counts
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()

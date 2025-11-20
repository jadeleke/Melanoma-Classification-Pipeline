"""
Training script with mixed precision, gradient accumulation, and early stopping.
Optimized for RTX 5060 8GB VRAM.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from tqdm import tqdm
import time
from pathlib import Path

from src.config import CONFIG
from src.model import create_model
from src.loss import create_loss_fn
from src.metrics import MetricsCalculator


class Trainer:
    """
    Training manager with all optimizations for 8GB VRAM.
    
    Features:
    - Mixed precision training (FP16)
    - Gradient accumulation (effective batch size = 32)
    - Gradient clipping
    - Early stopping
    - Learning rate scheduling with warmup
    - Automatic checkpointing
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
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler (with warmup)
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config['epochs'] - config['warmup_epochs'],
            T_mult=1,
            eta_min=1e-6
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config['mixed_precision'] else None
        
        # Tracking
        self.current_epoch = 0
        self.best_val_metric = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.train_losses = []
        self.val_losses = []
        self.val_metrics_history = []
        
        # Directories
        from pathlib import Path
        self.checkpoint_dir = Path("checkpoints")
        self.log_dir = Path("logs")
        self.results_dir = Path("results")
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
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
            
            # Mixed precision forward pass
            if self.config['mixed_precision']:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                    loss = loss / self.config['gradient_accumulation']
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss = loss / self.config['gradient_accumulation']
                loss.backward()
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.config['gradient_accumulation'] == 0:
                if self.config['mixed_precision']:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['gradient_clip']
                    )
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['gradient_clip']
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Track loss
            running_loss += loss.item() * self.config['gradient_accumulation']
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{running_loss / (batch_idx + 1):.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        epoch_loss = running_loss / num_batches
        self.train_losses.append(epoch_loss)
        
        return epoch_loss
    
    @torch.no_grad()
    def validate(self):
        """Validate on validation set."""
        self.model.eval()
        running_loss = 0.0
        num_batches = len(self.val_loader)
        
        # Metrics calculator
        metrics_calc = MetricsCalculator()
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Forward pass
            if self.config['mixed_precision']:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected in validation batch {len(self.val_losses)}")
                continue
            
            running_loss += loss.item()
            
            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # Update metrics
            metrics_calc.update(preds, probs, targets)
        
        # Calculate metrics
        epoch_loss = running_loss / num_batches if num_batches > 0 else float('inf')
        metrics = metrics_calc.compute()
        
        self.val_losses.append(epoch_loss)
        self.val_metrics_history.append(metrics)
        
        # Store last validation data for visualization
        self._last_val_targets = np.array(metrics_calc.all_targets)
        self._last_val_predictions = np.array(metrics_calc.all_preds)
        self._last_val_probabilities = np.array(metrics_calc.all_probs)
        
        return epoch_loss, metrics
    
    def train(self):
        """Full training loop."""
        print(f"\nStarting training on {self.device}")
        print(f"Model: {self.config['model_name']}")
        print(f"Epochs: {self.config['epochs']}")
        print(f"Batch size: {self.config['batch_size']} (effective: {self.config['batch_size'] * self.config['gradient_accumulation']})")
        print(f"Mixed precision: {self.config['mixed_precision']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print("="*60)
        
        start_time = time.time()
        
        for epoch in range(1, self.config['epochs'] + 1):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_metrics = self.validate()
            
            # Learning rate scheduling (after warmup)
            if epoch > self.config['warmup_epochs']:
                self.scheduler.step()
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{self.config['epochs']}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}" if not np.isnan(val_loss) and not np.isinf(val_loss) else f"  Val Loss:   {val_loss}")
            print(f"  Accuracy:   {val_metrics['accuracy']:.4f}")
            print(f"  AUC-ROC:    {val_metrics['auc_roc']:.4f}")
            print(f"  Sensitivity: {val_metrics['sensitivity']:.4f}")
            print(f"  Specificity: {val_metrics['specificity']:.4f}")
            
            # Check for improvement (using AUC-ROC as main metric)
            current_metric = val_metrics['auc_roc']
            
            if current_metric > self.best_val_metric:
                self.best_val_metric = current_metric
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                
                # Save best model
                self.save_checkpoint('best_model.pth', is_best=True)
                print(f"  ✓ New best model! (AUC-ROC: {current_metric:.4f})")
            else:
                self.epochs_without_improvement += 1
                print(f"  No improvement for {self.epochs_without_improvement} epochs")
            
            # Early stopping
            if self.epochs_without_improvement >= self.config['early_stopping_patience']:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
            
            # Save last checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
        
        # Save final training logs
        self.save_training_logs()
        
        # Training complete
        elapsed_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {elapsed_time/3600:.2f} hours")
        print(f"Best validation AUC-ROC: {self.best_val_metric:.4f} (epoch {self.best_epoch})")
        print(f"{'='*60}\n")
        
        # Create training visualizations
        self.create_training_visualizations(elapsed_time/3600)
        
        return self.best_val_metric
    
    def create_training_visualizations(self, training_time_hours):
        """Create comprehensive training visualizations."""
        print(f"\n{'='*60}")
        print("CREATING TRAINING VISUALIZATIONS")
        print(f"{'='*60}")
        
        try:
            from src.visualizer import TrainingVisualizer
            
            # Collect all predictions and targets from last validation
            if hasattr(self, '_last_val_targets') and hasattr(self, '_last_val_predictions'):
                visualizer = TrainingVisualizer(self.results_dir)
                visualizer.create_summary_dashboard(
                    self._last_val_targets,
                    self._last_val_predictions, 
                    self._last_val_probabilities,
                    self.train_losses,
                    self.val_losses,
                    self.val_metrics_history,
                    self.config,
                    training_time_hours
                )
            else:
                print("Warning: No validation data available for visualization")
                
        except Exception as e:
            print(f"Warning: Could not create visualizations: {e}")
    
    def save_training_logs(self):
        """Save training logs to file."""
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'training_progress': {
                'epochs_completed': self.current_epoch,
                'best_epoch': self.best_epoch,
                'best_val_metric': self.best_val_metric,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_metrics_history': self.val_metrics_history
            }
        }
        
        log_path = self.log_dir / f"training_log_{timestamp}.json"
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        print(f"Training logs saved to: {log_path}")
    
    def save_checkpoint(self, filename, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_metric': self.best_val_metric,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics_history': self.val_metrics_history,
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        
        if is_best:
            print(f"  Saved best checkpoint: {save_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint for resuming training."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_metric = checkpoint['best_val_metric']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")


if __name__ == "__main__":
    print("Training module ready")
    print("Use train_model.py to start training")

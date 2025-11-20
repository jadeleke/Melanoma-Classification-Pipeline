"""
Training visualization and analysis tools.
Creates comprehensive plots and reports for training results.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report
import json
from datetime import datetime


class TrainingVisualizer:
    """
    Comprehensive visualization suite for training results.
    """
    
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_training_curves(self, train_losses, val_losses, val_metrics_history, save_prefix="training"):
        """Plot training and validation curves."""
        epochs = range(1, len(train_losses) + 1)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        # Loss curves
        axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        if len(val_losses) == len(train_losses):
            val_losses_clean = [loss for loss in val_losses if not (np.isnan(loss) or np.isinf(loss))]
            if val_losses_clean:
                axes[0, 0].plot(epochs[:len(val_losses_clean)], val_losses_clean, 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Extract metrics over epochs
        metrics_names = ['accuracy', 'auc_roc', 'sensitivity', 'specificity', 'precision', 'f1']
        
        for i, metric in enumerate(metrics_names):
            row = (i + 1) // 3
            col = (i + 1) % 3
            
            if row >= 2:  # Skip if we don't have enough subplots
                break
                
            metric_values = [m.get(metric, 0) for m in val_metrics_history]
            axes[row, col].plot(epochs, metric_values, 'g-', linewidth=2, marker='o', markersize=4)
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].set_ylabel(metric.replace('_', ' ').title())
            axes[row, col].set_title(f'Validation {metric.replace("_", " ").title()}')
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].set_ylim(0, 1)
        
        plt.tight_layout()
        save_path = self.results_dir / f"{save_prefix}_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to: {save_path}")
    
    def plot_final_metrics(self, metrics, save_prefix="final"):
        """Plot final validation metrics as bar chart."""
        metric_names = ['accuracy', 'precision', 'recall', 'sensitivity', 'specificity', 'f1', 'auc_roc', 'auc_pr']
        metric_values = [metrics.get(m, 0) for m in metric_names]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(range(len(metric_names)), metric_values, 
                     color=sns.color_palette("viridis", len(metric_names)))
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, metric_values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Final Validation Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(metric_names)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metric_names], rotation=45)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.results_dir / f"{save_prefix}_metrics.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Final metrics plot saved to: {save_path}")
    
    def plot_confusion_matrix(self, targets, predictions, save_prefix="confusion"):
        """Plot confusion matrix."""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Benign', 'Malignant'],
                   yticklabels=['Benign', 'Malignant'],
                   cbar_kws={'label': 'Count'})
        
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        # Add accuracy information
        accuracy = np.trace(cm) / np.sum(cm)
        plt.figtext(0.02, 0.02, f'Accuracy: {accuracy:.4f}', fontsize=10, bbox=dict(boxstyle="round", facecolor='wheat'))
        
        save_path = self.results_dir / f"{save_prefix}_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to: {save_path}")
    
    def plot_roc_curves(self, targets, probabilities, save_prefix="roc"):
        """Plot ROC curve."""
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(targets, probabilities)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        save_path = self.results_dir / f"{save_prefix}_curve.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ROC curve saved to: {save_path}")
    
    def plot_precision_recall_curve(self, targets, probabilities, save_prefix="pr"):
        """Plot Precision-Recall curve."""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        precision, recall, _ = precision_recall_curve(targets, probabilities)
        avg_precision = average_precision_score(targets, probabilities)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='darkgreen', lw=2, label=f'PR curve (AP = {avg_precision:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        save_path = self.results_dir / f"{save_prefix}_curve.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Precision-Recall curve saved to: {save_path}")
    
    def save_detailed_report(self, targets, predictions, probabilities, 
                           train_losses, val_losses, val_metrics_history, 
                           config, training_time, save_prefix="report"):
        """Save comprehensive training report."""
        from sklearn.metrics import classification_report
        
        # Create detailed report
        report = {
            'timestamp': datetime.now().isoformat(),
            'training_time_hours': training_time,
            'config': config,
            'final_metrics': val_metrics_history[-1] if val_metrics_history else {},
            'training_summary': {
                'total_epochs': len(train_losses),
                'final_train_loss': train_losses[-1] if train_losses else None,
                'final_val_loss': val_losses[-1] if val_losses else None,
                'best_auc_roc': max([m.get('auc_roc', 0) for m in val_metrics_history]) if val_metrics_history else 0,
            },
            'classification_report': classification_report(targets, predictions, output_dict=True),
        }
        
        # Save JSON report
        json_path = self.results_dir / f"{save_prefix}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save text report
        text_path = self.results_dir / f"{save_prefix}.txt"
        with open(text_path, 'w') as f:
            f.write("MELANOMA CLASSIFICATION TRAINING REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {report['timestamp']}\n")
            f.write(f"Training Time: {training_time:.2f} hours\n")
            f.write(f"Total Epochs: {len(train_losses)}\n\n")
            
            f.write("CONFIGURATION:\n")
            f.write("-" * 20 + "\n")
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write("FINAL METRICS:\n")
            f.write("-" * 20 + "\n")
            if val_metrics_history:
                final_metrics = val_metrics_history[-1]
                for key, value in final_metrics.items():
                    if isinstance(value, (int, float)):
                        f.write(f"{key}: {value:.4f}\n")
            f.write("\n")
            
            f.write("CLASSIFICATION REPORT:\n")
            f.write("-" * 20 + "\n")
            f.write(classification_report(targets, predictions))
        
        print(f"Detailed reports saved to: {json_path} and {text_path}")
    
    def create_summary_dashboard(self, targets, predictions, probabilities,
                               train_losses, val_losses, val_metrics_history,
                               config, training_time):
        """Create a comprehensive dashboard with all visualizations."""
        print(f"\n{'='*60}")
        print("CREATING TRAINING VISUALIZATION DASHBOARD")
        print(f"{'='*60}")
        
        # Generate timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create all plots
        self.plot_training_curves(train_losses, val_losses, val_metrics_history, f"training_{timestamp}")
        self.plot_final_metrics(val_metrics_history[-1] if val_metrics_history else {}, f"metrics_{timestamp}")
        self.plot_confusion_matrix(targets, predictions, f"confusion_{timestamp}")
        
        # Only create ROC/PR curves if we have valid probabilities
        if len(np.unique(targets)) > 1 and len(probabilities) > 0:
            try:
                self.plot_roc_curves(targets, probabilities, f"roc_{timestamp}")
                self.plot_precision_recall_curve(targets, probabilities, f"pr_{timestamp}")
            except Exception as e:
                print(f"Warning: Could not create ROC/PR curves: {e}")
        
        # Save detailed report
        self.save_detailed_report(targets, predictions, probabilities,
                                train_losses, val_losses, val_metrics_history,
                                config, training_time, f"report_{timestamp}")
        
        print(f"\n{'='*60}")
        print(f"DASHBOARD COMPLETE! All files saved to: {self.results_dir}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test visualization
    print("Testing training visualizer...")
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 1000
    
    targets = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    probabilities = np.random.beta(2, 8, size=n_samples)
    predictions = (probabilities > 0.5).astype(int)
    
    # Dummy training history
    train_losses = [0.8, 0.6, 0.4, 0.3, 0.25]
    val_losses = [0.7, 0.5, 0.45, 0.4, 0.35]
    val_metrics_history = [
        {'accuracy': 0.85, 'auc_roc': 0.82, 'sensitivity': 0.8, 'specificity': 0.85, 'precision': 0.7, 'f1': 0.75},
        {'accuracy': 0.87, 'auc_roc': 0.85, 'sensitivity': 0.82, 'specificity': 0.87, 'precision': 0.72, 'f1': 0.77},
        {'accuracy': 0.89, 'auc_roc': 0.87, 'sensitivity': 0.84, 'specificity': 0.89, 'precision': 0.74, 'f1': 0.79},
        {'accuracy': 0.91, 'auc_roc': 0.89, 'sensitivity': 0.86, 'specificity': 0.91, 'precision': 0.76, 'f1': 0.81},
        {'accuracy': 0.92, 'auc_roc': 0.90, 'sensitivity': 0.87, 'specificity': 0.92, 'precision': 0.78, 'f1': 0.82},
    ]
    
    config = {'model_name': 'efficientnet_b3', 'epochs': 5, 'batch_size': 8}
    
    # Create visualizer and dashboard
    visualizer = TrainingVisualizer("test_results")
    visualizer.create_summary_dashboard(targets, predictions, probabilities,
                                      train_losses, val_losses, val_metrics_history,
                                      config, 1.5)
    
    print("✓ Visualization test complete")
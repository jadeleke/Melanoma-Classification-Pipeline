"""
Model comparison script: Baseline vs Hybrid model evaluation.
Compares performance, fusion weights, and component contributions.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
import json

from src.config import CONFIG
from src.models.efficientnet import create_model as create_baseline_model
from src.models.hybrid_model import create_hybrid_model
from src.dataset import create_dataloaders
from src.metrics import MetricsCalculator


class ModelComparator:
    """
    Compare baseline and hybrid models.
    
    Generates:
    - Side-by-side metric comparison
    - ROC curve comparison
    - Fusion weight analysis
    - Component ablation study
    """
    
    def __init__(self, baseline_checkpoint, hybrid_checkpoint, config):
        self.baseline_checkpoint = baseline_checkpoint
        self.hybrid_checkpoint = hybrid_checkpoint
        self.config = config
        self.device = config['device']
        
        # Create output directory
        self.comparison_dir = Path("results/comparison")
        self.comparison_dir.mkdir(parents=True, exist_ok=True)
    
    def load_models(self):
        """Load both baseline and hybrid models."""
        print("Loading models...")
        
        # Baseline model
        self.baseline_model = create_baseline_model(self.config)
        checkpoint = torch.load(self.baseline_checkpoint, map_location=self.device)
        self.baseline_model.load_state_dict(checkpoint['model_state_dict'])
        self.baseline_model.to(self.device)
        self.baseline_model.eval()
        
        # Hybrid model
        self.hybrid_model = create_hybrid_model(self.config)
        checkpoint = torch.load(self.hybrid_checkpoint, map_location=self.device)
        self.hybrid_model.load_state_dict(checkpoint['model_state_dict'])
        self.hybrid_model.to(self.device)
        self.hybrid_model.eval()
        
        print("✓ Models loaded successfully")
    
    def evaluate_model(self, model, data_loader, model_name="Model"):
        """Evaluate a single model."""
        print(f"\nEvaluating {model_name}...")
        
        metrics_calc = MetricsCalculator()
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                images = batch['image'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # Forward pass
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                # Update metrics
                metrics_calc.update(preds, probs, targets)
                
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Compute metrics
        metrics = metrics_calc.compute()
        
        return metrics, np.array(all_probs), np.array(all_targets)
    
    def compare_metrics(self, baseline_metrics, hybrid_metrics):
        """Create comparison table and visualization."""
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        
        # Metrics to compare
        metric_names = ['accuracy', 'sensitivity', 'specificity', 'f1', 'auc_roc', 'auc_pr']
        
        print(f"{'Metric':<20} {'Baseline':>12} {'Hybrid':>12} {'Improvement':>15}")
        print("-" * 80)
        
        improvements = {}
        for metric in metric_names:
            baseline_val = baseline_metrics.get(metric, 0)
            hybrid_val = hybrid_metrics.get(metric, 0)
            improvement = ((hybrid_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0
            improvements[metric] = improvement
            
            print(f"{metric.replace('_', ' ').title():<20} {baseline_val:>12.4f} {hybrid_val:>12.4f} {improvement:>+14.2f}%")
        
        print("="*80)
        
        # Save comparison
        comparison_data = {
            'baseline': {k: float(v) for k, v in baseline_metrics.items() if k in metric_names},
            'hybrid': {k: float(v) for k, v in hybrid_metrics.items() if k in metric_names},
            'improvements': improvements
        }
        
        with open(self.comparison_dir / "comparison.json", 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        # Visualization
        self._plot_metric_comparison(baseline_metrics, hybrid_metrics, metric_names)
    
    def _plot_metric_comparison(self, baseline_metrics, hybrid_metrics, metric_names):
        """Plot side-by-side metric comparison."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        baseline_vals = [baseline_metrics.get(m, 0) for m in metric_names]
        hybrid_vals = [hybrid_metrics.get(m, 0) for m in metric_names]
        
        ax.bar(x - width/2, baseline_vals, width, label='Baseline', alpha=0.8)
        ax.bar(x + width/2, hybrid_vals, width, label='Hybrid', alpha=0.8)
        
        ax.set_xlabel('Metric')
        ax.set_ylabel('Score')
        ax.set_title('Baseline vs Hybrid Model Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metric_names], rotation=45, ha='right')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(self.comparison_dir / "metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Saved comparison plot: {self.comparison_dir / 'metrics_comparison.png'}")
    
    def compare_roc_curves(self, baseline_probs, baseline_targets, hybrid_probs, hybrid_targets):
        """Plot ROC curves for both models."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Baseline ROC
        baseline_fpr, baseline_tpr, _ = roc_curve(baseline_targets, baseline_probs)
        baseline_auc = roc_auc_score(baseline_targets, baseline_probs)
        
        # Hybrid ROC
        hybrid_fpr, hybrid_tpr, _ = roc_curve(hybrid_targets, hybrid_probs)
        hybrid_auc = roc_auc_score(hybrid_targets, hybrid_probs)
        
        # Plot
        ax.plot(baseline_fpr, baseline_tpr, linewidth=2, label=f'Baseline (AUC = {baseline_auc:.4f})')
        ax.plot(hybrid_fpr, hybrid_tpr, linewidth=2, label=f'Hybrid (AUC = {hybrid_auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve Comparison', fontsize=14)
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.comparison_dir / "roc_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved ROC comparison: {self.comparison_dir / 'roc_comparison.png'}")
    
    def analyze_fusion_weights(self):
        """Analyze fusion weights from hybrid model."""
        if not hasattr(self.hybrid_model, 'get_feature_importance'):
            print("\n⚠️  Hybrid model does not have fusion weights")
            return
        
        print("\n" + "="*60)
        print("FUSION WEIGHT ANALYSIS")
        print("="*60)
        
        importance = self.hybrid_model.get_feature_importance()
        
        print("\nComponent Importance:")
        for component, weight in importance.items():
            print(f"  {component:15s}: {weight:.4f} ({weight*100:.1f}%)")
        
        print("="*60)
        
        # Visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        
        components = list(importance.keys())
        weights = list(importance.values())
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(components)))
        
        ax.bar(components, weights, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Fusion Weight', fontsize=12)
        ax.set_title('Feature Fusion Weights (WOA-Optimized)', fontsize=14)
        ax.set_ylim([0, 1])
        ax.grid(alpha=0.3, axis='y')
        
        # Add percentage labels
        for i, (comp, weight) in enumerate(zip(components, weights)):
            ax.text(i, weight + 0.02, f'{weight*100:.1f}%', ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(self.comparison_dir / "fusion_weights.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Saved fusion weights plot: {self.comparison_dir / 'fusion_weights.png'}")
    
    def run_comparison(self, val_csv, img_dir):
        """Run complete comparison."""
        print("\n" + "="*80)
        print("MODEL COMPARISON SUITE")
        print("="*80)
        
        # Load models
        self.load_models()
        
        # Create data loader
        from src.dataset import MelanomaDataset, get_val_transforms
        from torch.utils.data import DataLoader
        
        print("\nLoading validation data...")
        val_dataset = MelanomaDataset(
            csv_path=val_csv,
            img_dir=img_dir,
            image_size=self.config['image_size'],
            transform=get_val_transforms(
                image_size=self.config['image_size'],
                mean=self.config['mean'],
                std=self.config['std']
            ),
            mode='val'
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'] * 2,
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True if self.config['device'] == 'cuda' else False
        )
        
        # Evaluate both models
        baseline_metrics, baseline_probs, baseline_targets = self.evaluate_model(
            self.baseline_model, val_loader, "Baseline"
        )
        
        hybrid_metrics, hybrid_probs, hybrid_targets = self.evaluate_model(
            self.hybrid_model, val_loader, "Hybrid"
        )
        
        # Compare metrics
        self.compare_metrics(baseline_metrics, hybrid_metrics)
        
        # Compare ROC curves
        self.compare_roc_curves(
            baseline_probs, baseline_targets,
            hybrid_probs, hybrid_targets
        )
        
        # Analyze fusion weights
        self.analyze_fusion_weights()
        
        print("\n" + "="*80)
        print("COMPARISON COMPLETE")
        print("="*80)
        print(f"Results saved to: {self.comparison_dir}")
        print("="*80 + "\n")


def main():
    """Main comparison function."""
    from src.config import CONFIG, DATA_DIR
    
    # Checkpoint paths
    baseline_checkpoint = "checkpoints/baseline_best_model.pth"
    hybrid_checkpoint = "checkpoints/best_model.pth"
    
    # Validation data
    val_csv = DATA_DIR / "train" / "val_split.csv"
    img_dir = DATA_DIR / "train" / "images"
    
    # Check if checkpoints exist
    if not Path(baseline_checkpoint).exists():
        print(f"⚠️  Baseline checkpoint not found: {baseline_checkpoint}")
        print("   Please train the baseline model first using: python train_enhanced.py")
        return
    
    if not Path(hybrid_checkpoint).exists():
        print(f"⚠️  Hybrid checkpoint not found: {hybrid_checkpoint}")
        print("   Please train the hybrid model first using: python train_hybrid.py")
        return
    
    # Run comparison
    comparator = ModelComparator(baseline_checkpoint, hybrid_checkpoint, CONFIG)
    comparator.run_comparison(val_csv, img_dir)


if __name__ == "__main__":
    main()

"""
Evaluation metrics for melanoma classification.
Implements accuracy, sensitivity, specificity, AUC-ROC, AUC-PR, F1-score.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsCalculator:
    """
    Calculate and track evaluation metrics.
    Extended for hybrid model with component contribution analysis.
    
    Metrics:
    - Accuracy: Overall correctness
    - Sensitivity (Recall): True positive rate
    - Specificity: True negative rate
    - Precision: Positive predictive value
    - F1-Score: Harmonic mean of precision and recall
    - AUC-ROC: Area under ROC curve
    - AUC-PR: Area under precision-recall curve
    
    Hybrid Model Metrics:
    - Fusion weight tracking
    - Component importance
    """
    
    def __init__(self, track_fusion_weights=False):
        self.track_fusion_weights = track_fusion_weights
        self.fusion_weight_history = [] if track_fusion_weights else None
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.all_preds = []
        self.all_probs = []
        self.all_targets = []
        if self.fusion_weight_history is not None:
            self.fusion_weight_history = []
    
    def update(self, preds, probs, targets):
        """
        Update metrics with new batch.
        
        Args:
            preds: (batch_size,) predicted classes
            probs: (batch_size, num_classes) prediction probabilities
            targets: (batch_size,) ground truth labels
        """
        # Convert to numpy
        if torch.is_tensor(preds):
            preds = preds.cpu().numpy()
        if torch.is_tensor(probs):
            probs = probs.cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.cpu().numpy()
        
        self.all_preds.extend(preds)
        self.all_probs.extend(probs[:, 1])  # Probability of positive class
        self.all_targets.extend(targets)
    
    def compute(self):
        """
        Compute all metrics.
        
        Returns:
            dict: Dictionary of metric values
        """
        preds = np.array(self.all_preds)
        probs = np.array(self.all_probs)
        targets = np.array(self.all_targets)
        
        # Calculate metrics
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(targets, preds)
        metrics['precision'] = precision_score(targets, preds, zero_division=0)
        metrics['recall'] = recall_score(targets, preds, zero_division=0)
        metrics['sensitivity'] = metrics['recall']  # Same as recall
        metrics['f1'] = f1_score(targets, preds, zero_division=0)
        
        # Specificity (True Negative Rate)
        tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # AUC metrics (require probabilities)
        try:
            # Check if we have both classes in targets
            unique_targets = np.unique(targets)
            if len(unique_targets) < 2:
                print(f"Warning: Only one class present in targets: {unique_targets}")
                metrics['auc_roc'] = 0.0
                metrics['auc_pr'] = 0.0
            elif len(probs) == 0:
                print("Warning: No probability scores available")
                metrics['auc_roc'] = 0.0
                metrics['auc_pr'] = 0.0
            else:
                # Check for valid probability range
                if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
                    print("Warning: NaN or Inf in probability scores")
                    metrics['auc_roc'] = 0.0
                    metrics['auc_pr'] = 0.0
                else:
                    metrics['auc_roc'] = roc_auc_score(targets, probs)
                    metrics['auc_pr'] = average_precision_score(targets, probs)
        except Exception as e:
            print(f"Warning: AUC calculation failed: {e}")
            metrics['auc_roc'] = 0.0
            metrics['auc_pr'] = 0.0
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(targets, preds)
        metrics['tn'] = tn
        metrics['fp'] = fp
        metrics['fn'] = fn
        metrics['tp'] = tp
        
        return metrics
    
    def plot_confusion_matrix(self, save_path=None):
        """Plot confusion matrix."""
        metrics = self.compute()
        cm = metrics['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Benign', 'Malignant'],
                   yticklabels=['Benign', 'Malignant'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    
    def plot_roc_curve(self, save_path=None):
        """Plot ROC curve."""
        targets = np.array(self.all_targets)
        probs = np.array(self.all_probs)
        
        fpr, tpr, _ = roc_curve(targets, probs)
        auc = roc_auc_score(targets, probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    
    def plot_pr_curve(self, save_path=None):
        """Plot Precision-Recall curve."""
        targets = np.array(self.all_targets)
        probs = np.array(self.all_probs)
        
        precision, recall, _ = precision_recall_curve(targets, probs)
        auc_pr = average_precision_score(targets, probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label=f'PR (AUC = {auc_pr:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    
    def update_fusion_weights(self, fusion_weights):
        """
        Track fusion weights over time (for hybrid model).
        
        Args:
            fusion_weights: Dictionary or tensor of fusion weights
        """
        if self.fusion_weight_history is not None:
            if torch.is_tensor(fusion_weights):
                fusion_weights = fusion_weights.detach().cpu().numpy()
            self.fusion_weight_history.append(fusion_weights.copy() if isinstance(fusion_weights, np.ndarray) else fusion_weights)
    
    def get_fusion_weight_stats(self):
        """Get statistics of fusion weights over time."""
        if self.fusion_weight_history is None or len(self.fusion_weight_history) == 0:
            return None
        
        weights_array = np.array(self.fusion_weight_history)
        return {
            'mean': np.mean(weights_array, axis=0),
            'std': np.std(weights_array, axis=0),
            'min': np.min(weights_array, axis=0),
            'max': np.max(weights_array, axis=0),
            'final': weights_array[-1]
        }
    
    def print_metrics(self, fusion_weights=None):
        """Print all metrics in formatted table."""
        metrics = self.compute()
        
        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        print(f"Accuracy:     {metrics['accuracy']:.4f}")
        print(f"Precision:    {metrics['precision']:.4f}")
        print(f"Recall:       {metrics['recall']:.4f}")
        print(f"Sensitivity:  {metrics['sensitivity']:.4f}")
        print(f"Specificity:  {metrics['specificity']:.4f}")
        print(f"F1-Score:     {metrics['f1']:.4f}")
        print(f"AUC-ROC:      {metrics['auc_roc']:.4f}")
        print(f"AUC-PR:       {metrics['auc_pr']:.4f}")
        print("\nConfusion Matrix:")
        print(f"  TN: {metrics['tn']:5d}  |  FP: {metrics['fp']:5d}")
        print(f"  FN: {metrics['fn']:5d}  |  TP: {metrics['tp']:5d}")
        
        # Print fusion weights if available
        if fusion_weights is not None:
            print("\nFusion Weights:")
            if isinstance(fusion_weights, dict):
                for component, weight in fusion_weights.items():
                    print(f"  {component:15s}: {weight:.4f}")
            elif torch.is_tensor(fusion_weights):
                weights = torch.softmax(fusion_weights, dim=0).detach().cpu().numpy()
                print(f"  HRNet:          {weights[0]:.4f}")
                print(f"  EfficientNet:   {weights[1]:.4f}")
        
        print("="*50 + "\n")


if __name__ == "__main__":
    # Test metrics calculation
    print("Testing metrics implementation...")
    
    try:
        # Create dummy predictions
        np.random.seed(42)
        n_samples = 1000
        
        # Simulate predictions (90% benign, 10% malignant)
        targets = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
        probs_pos = np.random.beta(2, 5, size=n_samples)  # Probabilities for positive class
        probs = np.column_stack([1 - probs_pos, probs_pos])
        preds = (probs_pos > 0.5).astype(int)
        
        # Calculate metrics
        calculator = MetricsCalculator()
        calculator.update(preds, probs, targets)
        calculator.print_metrics()
        
        print("✓ Metrics calculation successful")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

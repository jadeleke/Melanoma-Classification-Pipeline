"""
Loss functions for melanoma classification.
Focal Loss to handle class imbalance (2-5% melanoma positive rate).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weighting factor for positive class (default: 0.25)
               Will be adjusted based on actual class distribution
        gamma: Focusing parameter (default: 2.0)
               Higher gamma reduces loss for well-classified examples
        label_smoothing: Label smoothing factor (default: 0.0)
    
    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) logits
            targets: (batch_size,) class labels
        
        Returns:
            loss: Scalar focal loss
        """
        # Get probabilities
        p = F.softmax(inputs, dim=1)
        p = torch.clamp(p, min=1e-8, max=1 - 1e-8)
        
        # Get class probabilities
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)
        p_t = torch.clamp(p_t, min=1e-8, max=1 - 1e-8)
        
        # Calculate focal loss
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        if self.alpha is not None:
            alpha_t = torch.ones_like(targets, dtype=torch.float)
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_weight = alpha_t * focal_weight
        
        loss = focal_weight * ce_loss
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=1e6)
        
        return loss.mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy with label smoothing.
    Can be used as alternative to Focal Loss.
    
    Args:
        smoothing: Label smoothing factor (default: 0.1)
    """
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) logits
            targets: (batch_size,) class labels
        
        Returns:
            loss: Scalar loss
        """
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Create smoothed labels
        num_classes = inputs.size(1)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.view(-1, 1), 1)
        targets_smooth = targets_one_hot * (1 - self.smoothing) + self.smoothing / num_classes
        
        loss = (-targets_smooth * log_probs).sum(dim=1).mean()
        
        return loss


def create_loss_fn(config, class_counts=None):
    """
    Create loss function from configuration.
    
    Args:
        config: Configuration dictionary
        class_counts: Optional class distribution [count_class_0, count_class_1]
                     Used to calculate optimal alpha for Focal Loss
    
    Returns:
        loss_fn: Loss function
    """
    # Calculate alpha from class distribution if provided
    alpha = config['focal_alpha']
    if class_counts is not None:
        # alpha = count_class_0 / (count_class_0 + count_class_1)
        # This weights the rare class more heavily
        total = sum(class_counts)
        alpha = class_counts[0] / total
        print(f"Calculated Focal Loss alpha from class distribution: {alpha:.4f}")
    
    loss_fn = FocalLoss(
        alpha=alpha,
        gamma=config['focal_gamma'],
        label_smoothing=config.get('label_smoothing', 0.0)
    )
    
    return loss_fn


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")
    
    try:
        # Create dummy data
        batch_size = 4
        num_classes = 2
        inputs = torch.randn(batch_size, num_classes)
        targets = torch.tensor([0, 1, 0, 1])
        
        # Test Focal Loss
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        loss = focal_loss(inputs, targets)
        print(f"✓ Focal Loss: {loss.item():.4f}")
        
        # Test with class imbalance (95% class 0, 5% class 1)
        class_counts = [9500, 500]
        from config import CONFIG
        loss_fn = create_loss_fn(CONFIG, class_counts)
        loss = loss_fn(inputs, targets)
        print(f"✓ Focal Loss (adjusted alpha): {loss.item():.4f}")
        
        # Test Label Smoothing CE
        ls_ce = LabelSmoothingCrossEntropy(smoothing=0.1)
        loss = ls_ce(inputs, targets)
        print(f"✓ Label Smoothing CE: {loss.item():.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

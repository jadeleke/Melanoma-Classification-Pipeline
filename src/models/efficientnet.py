"""
EfficientNet feature extractor for classification features.
Modified from original model.py to extract features only.
"""

import torch
import torch.nn as nn
import timm


class EfficientNetFeatureExtractor(nn.Module):
    """
    EfficientNet-based feature extractor for deep classification features.
    
    Modified from the original MelanomaClassifier to only extract features
    without the final classification layer.
    
    Args:
        model_name: Name of timm model (default: 'efficientnet_b3')
        pretrained: Use ImageNet pretrained weights
        feature_dim: Output feature dimension (default: 1536 for B3)
        dropout: Dropout probability (default: 0.3)
        freeze_backbone: Freeze backbone layers for faster training
    """
    
    def __init__(self, model_name='efficientnet_b3', pretrained=True, 
                 feature_dim=None, dropout=0.3, freeze_backbone=False):
        super().__init__()
        
        # Load pretrained EfficientNet
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove original classifier
        )
        
        # Get feature dimension from backbone
        self.backbone_features = self.backbone.num_features
        
        # Use custom feature dimension or default to backbone output
        if feature_dim is None:
            feature_dim = self.backbone_features
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Feature projection layer (optional)
        if feature_dim != self.backbone_features:
            self.projection = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(self.backbone_features, feature_dim),
                nn.BatchNorm1d(feature_dim, eps=1e-3),
                nn.ReLU(inplace=True)
            )
        else:
            self.projection = nn.Sequential(
                nn.Dropout(p=dropout)
            )
        
        self.feature_dim = feature_dim
        
    def forward(self, x):
        """
        Forward pass to extract features.
        
        Args:
            x: Input tensor (batch_size, 3, H, W)
        
        Returns:
            features: Feature vector (batch_size, feature_dim)
        """
        # Extract features from backbone
        features = self.backbone(x)
        features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Apply projection
        features = self.projection(features)
        features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return features
    
    def get_num_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MelanomaClassifier(nn.Module):
    """
    Complete EfficientNet-B3 based classifier for melanoma detection.
    (Kept for backward compatibility with existing checkpoints)
    
    Architecture:
    - Backbone: EfficientNet-B3 (pretrained on ImageNet)
    - Dropout: 0.3 for regularization
    - Classifier: Linear layer for 2-class classification
    
    Args:
        model_name: Name of timm model (default: 'efficientnet_b3')
        pretrained: Use ImageNet pretrained weights
        num_classes: Number of output classes (default: 2)
        dropout: Dropout probability (default: 0.3)
    """
    
    def __init__(self, model_name='efficientnet_b3', pretrained=True, num_classes=2, dropout=0.3):
        super().__init__()
        
        # Load pretrained EfficientNet-B3
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove original classifier
        )
        
        # Get feature dimension
        self.num_features = self.model.num_features
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.num_features, num_classes)
        )
        
    def forward(self, x):
        """Forward pass."""
        # Extract features from backbone
        features = self.model(x)
        
        # Classify
        output = self.classifier(features)
        
        return output
    
    def get_num_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_efficientnet_extractor(config):
    """
    Create EfficientNet feature extractor from configuration.
    
    Args:
        config: Configuration dictionary with efficientnet settings
    
    Returns:
        model: Initialized EfficientNet feature extractor
    """
    eff_config = config.get('efficientnet', {})
    
    model = EfficientNetFeatureExtractor(
        model_name=eff_config.get('model_name', 'efficientnet_b3'),
        pretrained=eff_config.get('pretrained', True),
        feature_dim=eff_config.get('feature_dim', None),
        dropout=eff_config.get('dropout', 0.3),
        freeze_backbone=eff_config.get('freeze_backbone', False)
    )
    
    return model


def create_model(config):
    """
    Create complete model from configuration.
    (Backward compatibility function)
    
    Args:
        config: Configuration dictionary
    
    Returns:
        model: Initialized model
    """
    model = MelanomaClassifier(
        model_name=config['model_name'],
        pretrained=config['pretrained'],
        num_classes=config['num_classes'],
        dropout=config['dropout']
    )
    
    return model


if __name__ == "__main__":
    # Test feature extractor
    print("Testing EfficientNet feature extractor...")
    
    try:
        model = EfficientNetFeatureExtractor(model_name='efficientnet_b3', pretrained=False)
        
        print(f"\n✓ EfficientNet feature extractor created")
        print(f"  Total parameters: {model.get_num_parameters():,}")
        print(f"  Backbone features: {model.backbone_features}")
        print(f"  Output feature dimension: {model.feature_dim}")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        print(f"\n✓ Forward pass successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

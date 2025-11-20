"""
Model architecture: EfficientNet-B3 with custom classifier.
Replaces JAEO-LeNet baseline for improved performance.
"""

import torch
import torch.nn as nn
import timm


class MelanomaClassifier(nn.Module):
    """
    EfficientNet-B3 based classifier for melanoma detection.
    
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


def create_model(config):
    """
    Create model from configuration.
    
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
    # Test model creation
    print("Testing model architecture...")
    
    try:
        from config import CONFIG
        
        model = create_model(CONFIG)
        
        print(f"\n✓ Model created: {CONFIG['model_name']}")
        print(f"  Total parameters: {model.get_num_parameters():,}")
        print(f"  Feature dimension: {model.num_features}")
        print(f"  Dropout: {CONFIG['dropout']}")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, CONFIG['image_size'], CONFIG['image_size'])
        output = model(dummy_input)
        print(f"\n✓ Forward pass successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

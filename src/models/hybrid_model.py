"""
WOA-based Hybrid Model combining HRNet, EfficientNet, and StyleGAN.
Optimized feature fusion using Whale Optimization Algorithm.
"""

import torch
import torch.nn as nn
from .hrnet import HRNetFeatureExtractor
from .efficientnet import EfficientNetFeatureExtractor
from .stylegan import StyleGANAugmenter


class WOAHybridClassifier(nn.Module):
    """
    Hybrid classification model with WOA-optimized feature fusion.
    
    Architecture:
        1. HRNet → High-resolution spatial features
        2. EfficientNet → Deep classification features
        3. WOA-optimized fusion → Weighted feature combination
        4. Classifier → Final malignant/benign prediction
    
    Args:
        config: Configuration dictionary containing settings for all components
        num_classes: Number of output classes (default: 2)
    """
    
    def __init__(self, config, num_classes=2):
        super().__init__()
        
        self.config = config
        self.num_classes = num_classes
        
        # Component 1: HRNet for spatial features
        hrnet_config = config.get('hrnet', {})
        self.hrnet = HRNetFeatureExtractor(
            model_name=hrnet_config.get('model_name', 'hrnet_w32'),
            pretrained=hrnet_config.get('pretrained', True),
            feature_dim=hrnet_config.get('feature_dim', 2048),
            freeze_backbone=hrnet_config.get('freeze_backbone', False)
        )
        
        # Component 2: EfficientNet for classification features
        eff_config = config.get('efficientnet', {})
        self.efficientnet = EfficientNetFeatureExtractor(
            model_name=eff_config.get('model_name', 'efficientnet_b3'),
            pretrained=eff_config.get('pretrained', True),
            feature_dim=eff_config.get('feature_dim', 1536),
            dropout=eff_config.get('dropout', 0.3),
            freeze_backbone=eff_config.get('freeze_backbone', False)
        )
        
        # Component 3: StyleGAN for augmentation (optional, used in training only)
        stylegan_config = config.get('stylegan', {})
        if stylegan_config.get('enabled', False):
            self.stylegan = StyleGANAugmenter(
                generator_path=stylegan_config.get('generator_path', None),
                device=config.get('device', 'cuda'),
                latent_dim=stylegan_config.get('latent_dim', 512),
                truncation_psi=stylegan_config.get('truncation_psi', 0.7)
            )
        else:
            self.stylegan = None
        
        # Feature dimensions
        self.hrnet_dim = self.hrnet.feature_dim
        self.eff_dim = self.efficientnet.feature_dim
        self.total_features = self.hrnet_dim + self.eff_dim
        
        # Fusion weights (optimized by WOA)
        # Initialize with equal weights
        fusion_config = config.get('fusion', {})
        initial_weights = fusion_config.get('initial_weights', [0.5, 0.5])
        self.fusion_weights = nn.Parameter(
            torch.tensor(initial_weights, dtype=torch.float32),
            requires_grad=fusion_config.get('learnable', True)
        )
        
        # Fusion strategy
        self.fusion_strategy = fusion_config.get('strategy', 'weighted_sum')  # 'weighted_sum' or 'concat'
        
        if self.fusion_strategy == 'weighted_sum':
            # Features must have same dimension for weighted sum
            if self.hrnet_dim != self.eff_dim:
                # Add projection layers to match dimensions
                target_dim = max(self.hrnet_dim, self.eff_dim)
                
                if self.hrnet_dim < target_dim:
                    self.hrnet_projection = nn.Linear(self.hrnet_dim, target_dim)
                else:
                    self.hrnet_projection = nn.Identity()
                
                if self.eff_dim < target_dim:
                    self.eff_projection = nn.Linear(self.eff_dim, target_dim)
                else:
                    self.eff_projection = nn.Identity()
                
                self.fused_dim = target_dim
            else:
                self.hrnet_projection = nn.Identity()
                self.eff_projection = nn.Identity()
                self.fused_dim = self.hrnet_dim
        else:
            # Concatenation strategy
            self.fused_dim = self.total_features
        
        # Classification head
        hidden_dim = fusion_config.get('hidden_dim', 512)
        dropout = fusion_config.get('dropout', 0.5)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass through hybrid model.
        
        Args:
            x: Input images (batch_size, 3, H, W)
        
        Returns:
            output: Classification logits (batch_size, num_classes)
        """
        # Extract features from both networks
        hrnet_features = self.hrnet(x)
        eff_features = self.efficientnet(x)
        
        # Normalize fusion weights
        normalized_weights = torch.softmax(self.fusion_weights, dim=0)
        
        # Fuse features based on strategy
        if self.fusion_strategy == 'weighted_sum':
            # Project to same dimension if needed
            hrnet_features = self.hrnet_projection(hrnet_features)
            eff_features = self.eff_projection(eff_features)
            # Numerical safety
            hrnet_features = torch.nan_to_num(hrnet_features, nan=0.0, posinf=1e6, neginf=-1e6)
            eff_features = torch.nan_to_num(eff_features, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Weighted sum fusion
            fused_features = (
                normalized_weights[0] * hrnet_features +
                normalized_weights[1] * eff_features
            )
        else:
            # Concatenation fusion
            fused_features = torch.cat([hrnet_features, eff_features], dim=1)
        
        # Final numerical safety before classification
        fused_features = torch.nan_to_num(fused_features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Final classification
        output = self.classifier(fused_features)
        output = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return output
    
    def get_feature_importance(self):
        """
        Get normalized fusion weights showing component importance.
        
        Returns:
            weights: Dictionary of component weights
        """
        normalized_weights = torch.softmax(self.fusion_weights, dim=0)
        
        return {
            'hrnet': normalized_weights[0].item(),
            'efficientnet': normalized_weights[1].item()
        }
    
    def get_num_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_component(self, component='hrnet'):
        """
        Freeze a specific component for faster training.
        
        Args:
            component: 'hrnet' or 'efficientnet'
        """
        if component == 'hrnet':
            for param in self.hrnet.parameters():
                param.requires_grad = False
        elif component == 'efficientnet':
            for param in self.efficientnet.parameters():
                param.requires_grad = False
    
    def unfreeze_component(self, component='hrnet'):
        """
        Unfreeze a specific component.
        
        Args:
            component: 'hrnet' or 'efficientnet'
        """
        if component == 'hrnet':
            for param in self.hrnet.parameters():
                param.requires_grad = True
        elif component == 'efficientnet':
            for param in self.efficientnet.parameters():
                param.requires_grad = True


def create_hybrid_model(config):
    """
    Create WOA hybrid model from configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        model: Initialized hybrid model
    """
    model = WOAHybridClassifier(
        config=config,
        num_classes=config.get('num_classes', 2)
    )
    
    return model


if __name__ == "__main__":
    # Test hybrid model
    print("Testing WOA Hybrid Model...")
    
    try:
        # Minimal config for testing
        test_config = {
            'device': 'cpu',
            'num_classes': 2,
            'hrnet': {
                'model_name': 'hrnet_w32',
                'pretrained': False,
                'feature_dim': 512,
                'freeze_backbone': False
            },
            'efficientnet': {
                'model_name': 'efficientnet_b3',
                'pretrained': False,
                'feature_dim': 512,
                'dropout': 0.3,
                'freeze_backbone': False
            },
            'stylegan': {
                'enabled': False
            },
            'fusion': {
                'strategy': 'weighted_sum',
                'initial_weights': [0.5, 0.5],
                'hidden_dim': 256,
                'dropout': 0.5,
                'learnable': True
            }
        }
        
        model = create_hybrid_model(test_config)
        
        print(f"\n✓ Hybrid model created successfully")
        print(f"  Total parameters: {model.get_num_parameters():,}")
        print(f"  HRNet features: {model.hrnet_dim}")
        print(f"  EfficientNet features: {model.eff_dim}")
        print(f"  Fused features: {model.fused_dim}")
        print(f"  Fusion strategy: {model.fusion_strategy}")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        
        print(f"\n✓ Forward pass successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        
        # Test feature importance
        importance = model.get_feature_importance()
        print(f"\n✓ Feature importance:")
        for component, weight in importance.items():
            print(f"  {component}: {weight:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

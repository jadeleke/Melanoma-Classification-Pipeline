"""
HRNet (High-Resolution Network) for spatial feature extraction.
Maintains high-resolution representations throughout the network.
"""

import torch
import torch.nn as nn
import timm


class HRNetFeatureExtractor(nn.Module):
    """
    HRNet-based feature extractor for high-resolution spatial features.
    
    HRNet maintains high-resolution representations by connecting high-to-low resolution
    convolutions in parallel, making it excellent for capturing fine-grained spatial details
    in skin lesion images.
    
    Args:
        model_name: HRNet variant ('hrnet_w32' or 'hrnet_w48')
        pretrained: Use ImageNet pretrained weights
        feature_dim: Output feature dimension (default: 2048)
        freeze_backbone: Freeze backbone for faster training
    """
    
    def __init__(self, model_name='hrnet_w32', pretrained=True, feature_dim=2048, freeze_backbone=False):
        super().__init__()
        
        # Load pretrained HRNet from timm
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3)  # Multi-scale features
            )
        except Exception as e:
            # Fallback to alternative HRNet naming
            print(f"Warning: Could not load {model_name}, trying alternative...")
            try:
                self.backbone = timm.create_model(
                    'hrnet_w32',
                    pretrained=pretrained,
                    features_only=True,
                    out_indices=(0, 1, 2, 3)
                )
            except:
                # If HRNet not available, use ResNet as fallback
                print("Warning: HRNet not available, using ResNet50 as fallback")
                self.backbone = timm.create_model(
                    'resnet50',
                    pretrained=pretrained,
                    features_only=True,
                    out_indices=(0, 1, 2, 3)
                )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get feature info
        feature_info = self.backbone.feature_info.channels()
        total_channels = sum(feature_info)
        
        # Adaptive pooling for each scale
        self.adaptive_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((1, 1)) for _ in feature_info
        ])
        
        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(total_channels, feature_dim),
            nn.BatchNorm1d(feature_dim, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.feature_dim = feature_dim
        
    def forward(self, x):
        """
        Forward pass with multi-scale feature extraction.
        
        Args:
            x: Input tensor (batch_size, 3, H, W)
        
        Returns:
            features: Fused feature vector (batch_size, feature_dim)
        """
        # Extract multi-scale features
        multi_scale_features = self.backbone(x)
        
        # Pool each scale to same size
        pooled_features = []
        for feat, pool in zip(multi_scale_features, self.adaptive_pools):
            pooled = pool(feat)
            pooled = pooled.flatten(1)
            pooled_features.append(pooled)
        
        # Concatenate all scales
        concatenated = torch.cat(pooled_features, dim=1)
        # Numerical safety
        concatenated = torch.nan_to_num(concatenated, nan=0.0, posinf=1e6, neginf=-1e6)

        # Fuse features
        fused_features = self.fusion(concatenated)
        fused_features = torch.nan_to_num(fused_features, nan=0.0, posinf=1e6, neginf=-1e6)

        return fused_features
    
    def get_num_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_hrnet(config):
    """
    Create HRNet feature extractor from configuration.
    
    Args:
        config: Configuration dictionary with hrnet settings
    
    Returns:
        model: Initialized HRNet feature extractor
    """
    hrnet_config = config.get('hrnet', {})
    
    model = HRNetFeatureExtractor(
        model_name=hrnet_config.get('model_name', 'hrnet_w32'),
        pretrained=hrnet_config.get('pretrained', True),
        feature_dim=hrnet_config.get('feature_dim', 2048),
        freeze_backbone=hrnet_config.get('freeze_backbone', False)
    )
    
    return model


if __name__ == "__main__":
    # Test HRNet feature extractor
    print("Testing HRNet feature extractor...")
    
    try:
        model = HRNetFeatureExtractor(model_name='hrnet_w32', pretrained=False)
        
        print(f"\n✓ HRNet created successfully")
        print(f"  Total parameters: {model.get_num_parameters():,}")
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

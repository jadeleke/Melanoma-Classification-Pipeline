"""
Hybrid model components for WOA-based skin lesion classification.

Components:
- HRNet: High-resolution spatial feature extraction
- EfficientNet: Deep classification features
- StyleGAN: Data augmentation (synthetic image generation)
- WOA: Whale Optimization Algorithm for feature fusion
- HybridModel: Combined architecture
"""

from .hrnet import HRNetFeatureExtractor
from .efficientnet import EfficientNetFeatureExtractor
from .stylegan import StyleGANAugmenter
from .woa_optimizer import WhaleOptimizer
from .hybrid_model import WOAHybridClassifier

__all__ = [
    'HRNetFeatureExtractor',
    'EfficientNetFeatureExtractor',
    'StyleGANAugmenter',
    'WhaleOptimizer',
    'WOAHybridClassifier',
]

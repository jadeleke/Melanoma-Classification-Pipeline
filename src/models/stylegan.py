"""
StyleGAN2-based data augmentation for synthetic lesion image generation.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class StyleGANAugmenter:
    """
    StyleGAN2-based augmentation for generating synthetic lesion images.
    
    This module can generate synthetic skin lesion images for data augmentation
    during training. The generator should be pre-trained or fine-tuned on the
    ISIC-2020 dataset.
    
    Args:
        generator_path: Path to pretrained StyleGAN2 generator checkpoint
        device: Device to run generation on
        latent_dim: Latent space dimension (default: 512)
        truncation_psi: Truncation psi for quality control (default: 0.7)
    """
    
    def __init__(self, generator_path=None, device='cuda', latent_dim=512, truncation_psi=0.7):
        self.device = device
        self.latent_dim = latent_dim
        self.truncation_psi = truncation_psi
        self.generator = None
        
        if generator_path is not None and Path(generator_path).exists():
            self.load_generator(generator_path)
        else:
            print("⚠️  StyleGAN generator not loaded. Using fallback augmentation.")
            print(f"   Generator path: {generator_path}")
            self.generator = None
    
    def load_generator(self, generator_path):
        """
        Load pretrained StyleGAN2 generator.
        
        Args:
            generator_path: Path to generator checkpoint
        """
        try:
            # Try to import stylegan2
            try:
                from stylegan2_pytorch import ModelLoader
                loader = ModelLoader(base_dir=str(Path(generator_path).parent))
                self.generator = loader.load_model(str(Path(generator_path).name))
                self.generator.eval()
                self.generator.to(self.device)
                print(f"✓ StyleGAN2 generator loaded from {generator_path}")
            except ImportError:
                print("⚠️  stylegan2-pytorch not installed. StyleGAN augmentation disabled.")
                print("   Install with: pip install stylegan2-pytorch")
                self.generator = None
        except Exception as e:
            print(f"⚠️  Could not load StyleGAN generator: {e}")
            self.generator = None
    
    def generate_batch(self, batch_size, image_size=256):
        """
        Generate a batch of synthetic images.
        
        Args:
            batch_size: Number of images to generate
            image_size: Target image size
        
        Returns:
            images: Generated images (batch_size, 3, H, W) in range [-1, 1]
        """
        if self.generator is None:
            # Return random noise as fallback
            return torch.randn(batch_size, 3, image_size, image_size, device=self.device)
        
        with torch.no_grad():
            # Sample latent codes
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            
            # Generate images
            try:
                images = self.generator(z, truncation_psi=self.truncation_psi)
                
                # Resize if needed
                if images.shape[-1] != image_size:
                    images = torch.nn.functional.interpolate(
                        images, 
                        size=(image_size, image_size), 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                return images
            except Exception as e:
                print(f"⚠️  Generation failed: {e}")
                return torch.randn(batch_size, 3, image_size, image_size, device=self.device)
    
    def augment_batch(self, real_images, synthetic_ratio=0.2):
        """
        Augment a batch of real images with synthetic ones.
        
        Args:
            real_images: Real images (batch_size, 3, H, W)
            synthetic_ratio: Ratio of synthetic images to add (0-1)
        
        Returns:
            augmented_images: Combined real + synthetic images
            is_synthetic: Boolean mask indicating synthetic images
        """
        if self.generator is None or synthetic_ratio <= 0:
            # No augmentation
            return real_images, torch.zeros(real_images.size(0), dtype=torch.bool)
        
        batch_size = real_images.size(0)
        n_synthetic = int(batch_size * synthetic_ratio)
        
        if n_synthetic == 0:
            return real_images, torch.zeros(batch_size, dtype=torch.bool)
        
        # Generate synthetic images
        synthetic_images = self.generate_batch(n_synthetic, real_images.size(-1))
        
        # Normalize synthetic images to match real images
        # Assuming real images are normalized with ImageNet stats
        synthetic_images = (synthetic_images + 1) / 2  # Convert from [-1, 1] to [0, 1]
        
        # Combine real and synthetic
        augmented_images = torch.cat([real_images, synthetic_images], dim=0)
        
        # Create mask
        is_synthetic = torch.cat([
            torch.zeros(batch_size, dtype=torch.bool),
            torch.ones(n_synthetic, dtype=torch.bool)
        ])
        
        # Shuffle
        shuffle_idx = torch.randperm(augmented_images.size(0))
        augmented_images = augmented_images[shuffle_idx]
        is_synthetic = is_synthetic[shuffle_idx]
        
        return augmented_images, is_synthetic
    
    def generate_with_style_mixing(self, batch_size, image_size=256, mixing_prob=0.5):
        """
        Generate images with style mixing for increased diversity.
        
        Args:
            batch_size: Number of images to generate
            image_size: Target image size
            mixing_prob: Probability of style mixing
        
        Returns:
            images: Generated images with style mixing
        """
        if self.generator is None:
            return self.generate_batch(batch_size, image_size)
        
        with torch.no_grad():
            # Sample two sets of latent codes for mixing
            z1 = torch.randn(batch_size, self.latent_dim, device=self.device)
            z2 = torch.randn(batch_size, self.latent_dim, device=self.device)
            
            # Decide which images use style mixing
            use_mixing = torch.rand(batch_size) < mixing_prob
            
            try:
                images = []
                for i in range(batch_size):
                    if use_mixing[i]:
                        # Use style mixing (if supported by generator)
                        img = self.generator(z1[i:i+1], z2[i:i+1], truncation_psi=self.truncation_psi)
                    else:
                        img = self.generator(z1[i:i+1], truncation_psi=self.truncation_psi)
                    images.append(img)
                
                images = torch.cat(images, dim=0)
                
                # Resize if needed
                if images.shape[-1] != image_size:
                    images = torch.nn.functional.interpolate(
                        images, 
                        size=(image_size, image_size), 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                return images
            except Exception as e:
                print(f"⚠️  Style mixing failed, using standard generation: {e}")
                return self.generate_batch(batch_size, image_size)


def create_stylegan_augmenter(config):
    """
    Create StyleGAN augmenter from configuration.
    
    Args:
        config: Configuration dictionary with stylegan settings
    
    Returns:
        augmenter: Initialized StyleGAN augmenter
    """
    stylegan_config = config.get('stylegan', {})
    
    augmenter = StyleGANAugmenter(
        generator_path=stylegan_config.get('generator_path', None),
        device=config.get('device', 'cuda'),
        latent_dim=stylegan_config.get('latent_dim', 512),
        truncation_psi=stylegan_config.get('truncation_psi', 0.7)
    )
    
    return augmenter


if __name__ == "__main__":
    # Test StyleGAN augmenter
    print("Testing StyleGAN augmenter...")
    
    try:
        augmenter = StyleGANAugmenter(generator_path=None, device='cpu')
        
        print(f"\n✓ StyleGAN augmenter created")
        
        # Test batch augmentation
        dummy_images = torch.randn(4, 3, 224, 224)
        augmented, is_synthetic = augmenter.augment_batch(dummy_images, synthetic_ratio=0.25)
        
        print(f"\n✓ Augmentation test successful")
        print(f"  Original batch size: {dummy_images.shape[0]}")
        print(f"  Augmented batch size: {augmented.shape[0]}")
        print(f"  Synthetic images: {is_synthetic.sum().item()}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

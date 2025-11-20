"""
Whale Optimization Algorithm (WOA) for feature fusion weight optimization.

WOA is a nature-inspired metaheuristic algorithm that mimics the hunting behavior
of humpback whales. It's used here to optimize the fusion weights for combining
features from HRNet and EfficientNet.
"""

import numpy as np
import torch
from tqdm import tqdm


class WhaleOptimizer:
    """
    Whale Optimization Algorithm for optimizing feature fusion weights.
    
    The algorithm optimizes weights to maximize validation performance (e.g., AUC-ROC)
    while balancing contributions from different model components.
    
    Args:
        n_whales: Population size (number of whales)
        max_iter: Maximum number of iterations
        dim: Dimension of search space (number of fusion weights)
        bounds: Search space bounds (default: [0, 1] for weights)
        verbose: Print optimization progress
    """
    
    def __init__(self, n_whales=30, max_iter=50, dim=2, bounds=(0, 1), verbose=True):
        self.n_whales = n_whales
        self.max_iter = max_iter
        self.dim = dim
        self.bounds = bounds
        self.verbose = verbose
        
        # Best solution tracking
        self.best_position = None
        self.best_score = -np.inf
        self.convergence_curve = []

    def _normalize(self, vec):
        """Normalize a weight vector to be non-negative and sum to 1, robust to NaNs/Infs."""
        v = np.array(vec, dtype=np.float64)
        # Replace NaN/Inf
        v = np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=0.0)
        # Clip to [0,1]
        v = np.clip(v, self.bounds[0], self.bounds[1])
        s = v.sum()
        if not np.isfinite(s) or s <= 1e-12:
            # Fallback to uniform
            v = np.ones_like(v, dtype=np.float64) / len(v)
        else:
            v = v / s
        return v
        
    def initialize_population(self):
        """
        Initialize whale population with random positions.
        Positions are normalized to sum to 1 (valid fusion weights).
        
        Returns:
            population: Initial whale positions (n_whales, dim)
        """
        # Random initialization
        population = np.random.uniform(
            self.bounds[0], 
            self.bounds[1], 
            (self.n_whales, self.dim)
        )
        # Normalize each whale robustly
        for i in range(self.n_whales):
            population[i] = self._normalize(population[i])
        
        return population
    
    def optimize(self, objective_fn, normalize_weights=True):
        """
        Run WOA optimization to find optimal fusion weights.
        
        Args:
            objective_fn: Function that takes weights and returns score (higher is better)
            normalize_weights: Normalize weights to sum to 1
        
        Returns:
            best_weights: Optimized fusion weights
            best_score: Best achieved score
            convergence_curve: Score history over iterations
        """
        # Initialize population
        population = self.initialize_population()
        
        # Evaluate initial population
        scores = np.array([
            objective_fn(self._normalize(whale) if normalize_weights else whale)
            for whale in population
        ])
        
        # Initialize best solution
        best_idx = np.argmax(scores)
        self.best_position = population[best_idx].copy()
        self.best_score = scores[best_idx]
        self.convergence_curve = [self.best_score]
        
        if self.verbose:
            print(f"\n🐋 Starting WOA optimization...")
            print(f"   Population: {self.n_whales} whales")
            print(f"   Dimensions: {self.dim}")
            print(f"   Max iterations: {self.max_iter}")
            print(f"   Initial best score: {self.best_score:.4f}")
        
        # Main optimization loop
        iterator = tqdm(range(self.max_iter), desc="WOA Optimization") if self.verbose else range(self.max_iter)
        
        for iteration in iterator:
            # Linearly decrease a from 2 to 0
            a = 2 - iteration * (2 / self.max_iter)
            
            # Update each whale position
            for i in range(self.n_whales):
                # Random parameters
                r = np.random.random()
                A = 2 * a * r - a
                C = 2 * r
                
                l = np.random.uniform(-1, 1)
                p = np.random.random()
                
                # Update position based on WOA equations
                if p < 0.5:
                    if abs(A) < 1:
                        # Encircling prey (exploitation)
                        D = abs(C * self.best_position - population[i])
                        population[i] = self.best_position - A * D
                    else:
                        # Search for prey (exploration)
                        random_whale = population[np.random.randint(self.n_whales)]
                        D = abs(C * random_whale - population[i])
                        population[i] = random_whale - A * D
                else:
                    # Spiral updating position
                    b = 1  # Spiral shape constant
                    D = abs(self.best_position - population[i])
                    population[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + self.best_position
                
                # Ensure bounds and robust normalization
                population[i] = np.clip(population[i], self.bounds[0], self.bounds[1])
                if normalize_weights:
                    population[i] = self._normalize(population[i])
                
                # Evaluate new position
                score = objective_fn(population[i])
                
                # Update best solution
                if score > self.best_score:
                    self.best_score = score
                    self.best_position = population[i].copy()
            
            # Record convergence
            self.convergence_curve.append(self.best_score)
            
            # Update progress bar
            if self.verbose and hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({'Best Score': f'{self.best_score:.4f}'})
        
        if self.verbose:
            print(f"\n✓ WOA optimization completed")
            print(f"   Final best score: {self.best_score:.4f}")
            print(f"   Optimal weights: {self.best_position}")
        
        return self.best_position, self.best_score, self.convergence_curve


class FusionWeightOptimizer:
    """
    Wrapper for WOA optimizer specifically for feature fusion weight optimization.
    
    This class integrates WOA with PyTorch models for optimizing fusion weights
    during training.
    
    Args:
        model: Hybrid model with fusion weights
        val_loader: Validation data loader
        device: Device for computation
        woa_config: WOA hyperparameters
    """
    
    def __init__(self, model, val_loader, device, woa_config=None):
        self.model = model
        self.val_loader = val_loader
        self.device = device
        
        # WOA configuration
        if woa_config is None:
            woa_config = {
                'n_whales': 20,
                'max_iter': 30,
                'verbose': True
            }
        
        self.woa_config = woa_config
        
        # Cached validation features for fast objective evaluation
        self._feature_cache = None  # dict with keys: hr, ef, targets

    def _build_feature_cache(self):
        """
        Precompute validation features for HRNet and EfficientNet branches (projected
        to the fused_dim) to dramatically accelerate WOA evaluation by avoiding
        re-running the heavy backbones for every whale/iteration.
        Only enabled for 'weighted_sum' fusion.
        """
        if getattr(self.model, 'fusion_strategy', 'weighted_sum') != 'weighted_sum':
            return None  # Not supported for concat strategy

        self.model.eval()
        hr_feats = []
        ef_feats = []
        targets_all = []

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                targets = batch['target'].to(self.device)

                # Extract raw features from both branches
                hr = self.model.hrnet(images)
                ef = self.model.efficientnet(images)

                # Project to fused_dim using model's projections
                hr = self.model.hrnet_projection(hr)
                ef = self.model.eff_projection(ef)

                # Numerical safety and move to CPU to reduce GPU memory pressure
                hr = torch.nan_to_num(hr, nan=0.0, posinf=1e6, neginf=-1e6).cpu()
                ef = torch.nan_to_num(ef, nan=0.0, posinf=1e6, neginf=-1e6).cpu()

                hr_feats.append(hr)
                ef_feats.append(ef)
                targets_all.append(targets.cpu())

        hr_tensor = torch.cat(hr_feats, dim=0)
        ef_tensor = torch.cat(ef_feats, dim=0)
        targets_tensor = torch.cat(targets_all, dim=0)

        self._feature_cache = {
            'hr': hr_tensor,
            'ef': ef_tensor,
            'targets': targets_tensor,
        }
        return self._feature_cache
        
    def evaluate_weights(self, weights):
        """
        Evaluate a set of fusion weights on validation set.
        
        Args:
            weights: Fusion weights to evaluate
        
        Returns:
            score: Validation metric (e.g., AUC-ROC)
        """
        # Fast path: use precomputed features if available and strategy is weighted_sum
        if self._feature_cache is None and getattr(self.model, 'fusion_strategy', 'weighted_sum') == 'weighted_sum':
            self._build_feature_cache()

        fusion_strategy = getattr(self.model, 'fusion_strategy', 'weighted_sum')

        if self._feature_cache is not None and fusion_strategy == 'weighted_sum':
            # Normalize weights like in the model, robust to NaN/Inf
            w = torch.tensor(weights, dtype=torch.float32)
            if not torch.isfinite(w).all():
                w = torch.ones_like(w) / w.numel()
            else:
                w = torch.softmax(w, dim=0)
            w0, w1 = float(w[0]), float(w[1])

            hr = self._feature_cache['hr']  # on CPU
            ef = self._feature_cache['ef']  # on CPU
            targets = self._feature_cache['targets'].numpy()

            # Fuse on CPU then run classifier on GPU in chunks (classifier is light)
            fused = w0 * hr + w1 * ef  # CPU tensor [N, D]

            probs_all = []
            self.model.eval()
            with torch.no_grad():
                bs = 4096  # big chunks since classifier is small
                for i in range(0, fused.shape[0], bs):
                    chunk = fused[i:i+bs].to(self.device, non_blocking=True)
                    logits = self.model.classifier(chunk)
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
                    probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu()
                    probs_all.append(probs)

            all_probs = torch.cat(probs_all).numpy()

            from sklearn.metrics import roc_auc_score
            try:
                score = roc_auc_score(targets, all_probs)
            except Exception:
                score = 0.0
            return score

        # Fallback: full forward pass through the model (slower)
        with torch.no_grad():
            if hasattr(self.model, 'fusion_weights'):
                self.model.fusion_weights.data = torch.tensor(weights, dtype=torch.float32).to(self.device)

        self.model.eval()
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                targets = batch['target'].to(self.device)
                outputs = self.model(images)
                outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e6, neginf=-1e6)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_probs.append(probs.cpu())
                all_targets.append(targets.cpu())

        from sklearn.metrics import roc_auc_score
        all_probs = torch.cat(all_probs).numpy()
        all_targets = torch.cat(all_targets).numpy()
        try:
            score = roc_auc_score(all_targets, all_probs)
        except Exception:
            score = 0.0
        return score
    
    def optimize(self):
        """
        Run WOA optimization to find optimal fusion weights.
        
        Returns:
            best_weights: Optimized fusion weights
            best_score: Best validation score achieved
        """
        # Get number of fusion weights
        n_components = len(self.model.fusion_weights) if hasattr(self.model, 'fusion_weights') else 2
        
        # Create WOA optimizer
        woa = WhaleOptimizer(
            n_whales=self.woa_config.get('n_whales', 20),
            max_iter=self.woa_config.get('max_iter', 30),
            dim=n_components,
            bounds=(0, 1),
            verbose=self.woa_config.get('verbose', True)
        )
        
        # Run optimization
        best_weights, best_score, convergence_curve = woa.optimize(
            objective_fn=self.evaluate_weights,
            normalize_weights=True
        )
        
        # Set final weights
        with torch.no_grad():
            if hasattr(self.model, 'fusion_weights'):
                self.model.fusion_weights.data = torch.tensor(best_weights, dtype=torch.float32).to(self.device)
        
        return best_weights, best_score, convergence_curve


if __name__ == "__main__":
    # Test WOA optimizer
    print("Testing Whale Optimization Algorithm...")
    
    # Simple test function (Sphere function)
    def sphere_function(x):
        """Minimize: sum(x^2). Optimal at x=[0,0,0]"""
        return -np.sum(x**2)  # Negative because WOA maximizes
    
    woa = WhaleOptimizer(n_whales=20, max_iter=30, dim=3, bounds=(-5, 5), verbose=True)
    best_pos, best_score, curve = woa.optimize(sphere_function, normalize_weights=False)
    
    print(f"\n✓ Test completed")
    print(f"   Best position: {best_pos}")
    print(f"   Best score: {best_score:.6f}")
    print(f"   Expected optimal: [0, 0, 0]")

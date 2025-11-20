# Implementation Details: Melanoma Classification Pipeline

This document explains the end-to-end methodology and the exact metrics used in this repository for SIIM-ISIC melanoma binary classification (benign vs malignant). It reflects the code currently present under `src/` and the training runners at the project root.

## Problem setting

- Task: Supervised 2-class image classification (malignant = 1, benign = 0)
- Data: SIIM-ISIC Melanoma Classification images in JPEG or DICOM format
- Outputs: Per-image logits → probabilities → predicted class, plus training/validation artifacts (plots, reports, checkpoints)

## Data and splits

- Directory structure expected by the code:
  - `data/train/images/` with training images
  - `data/train/train_split.csv` and `data/train/val_split.csv` with columns at minimum: `image_name`, `target`
- Use `prepare_data.py` to create stratified train/val CSVs from the Kaggle `train.csv`.
- DICOM support: `src/dataset.py` loads `.dcm` using pydicom and applies VOI LUT with `apply_voi_lut`, normalizes to 0–255, and repeats channel to RGB.

## Preprocessing and augmentation

Implemented in `src/dataset.py` via Albumentations. For training, the pipeline is:
- Resize to `image_size × image_size`
- HorizontalFlip (p=0.5)
- VerticalFlip (p=0.5)
- RandomRotate90 (p=0.5)
- ShiftScaleRotate (shift_limit 0.1, scale_limit 0.1, rotate_limit 45°, p=0.5)
- RandomBrightnessContrast (±0.2, p=0.5)
- HueSaturationValue (hue ±20, sat ±30, val ±20, p=0.3)
- RandomGamma (80–120, p=0.3)
- GaussianBlur (3–5 px, p=0.3)
- CoarseDropout (up to 8 holes, size up to 1/8 of image edge, p=0.3)
- Normalize (ImageNet mean/std)
- ToTensorV2 (HWC→CHW)

Validation/test transforms are limited to Resize + Normalize + ToTensorV2.

Notes:
- Optional synthetic augmentation hooks exist in `MelanomaDataset` (`stylegan_augmenter` + `synthetic_ratio`), but a generator implementation is not bundled; this feature is off by default.

## Model architecture

Defined in `src/model.py` as `MelanomaClassifier` using timm EfficientNet-B3:
- Backbone: EfficientNet-B3 with `num_classes=0` to return a feature vector of size `num_features`
- Head: `Dropout(p=0.3)` → `Linear(num_features, 2)`
- Input size: governed by `CONFIG['image_size']` (default 256)

The factory `create_model(CONFIG)` returns this classifier with options:
- `model_name` (default `efficientnet_b3`)
- `pretrained` (ImageNet weights)
- `dropout`
- `num_classes` (fixed at 2 for binary)

## Loss functions

Implemented in `src/loss.py`.

### Focal Loss (default)

The focal loss mitigates class imbalance by down-weighting easy examples.

Formula for a given sample with true class t and predicted probabilities p:

$$\mathrm{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \log(p_t)$$

- $p_t$ is the probability assigned to the true class t
- $\alpha_t \in [0,1]$ is a class weighting factor
- $\gamma \ge 0$ focuses training on hard examples (default $\gamma = 2$)

In this code:
- `alpha` defaults to `CONFIG['focal_alpha']` but is recalculated from class counts when provided: $\alpha = \frac{\text{count(class 0)}}{\text{count(0)} + \text{count(1)}}$. This places more weight on the minority class.
- `gamma` is taken from `CONFIG['focal_gamma']` (default 2.0)

### Label Smoothing Cross Entropy (optional)

Provided as an alternative in `LabelSmoothingCrossEntropy`:

$$\mathcal{L} = -\sum_{c=1}^{C} y_c^{\text{(smooth)}} \log p_c \quad\text{with}\quad y^{\text{(smooth)}} = (1-\epsilon)\,y^{\text{one-hot}} + \frac{\epsilon}{C}$$

- `smoothing = \epsilon` comes from `CONFIG['label_smoothing']`

The default training path uses Focal Loss via `create_loss_fn`.

## Optimization and training loop

Implemented in `src/train.py` (`Trainer`). Key aspects:
- Optimizer: AdamW with `learning_rate` and `weight_decay` from `CONFIG`
- Scheduler: CosineAnnealingWarmRestarts with `T_0 = epochs - warmup_epochs`
- Mixed precision: torch AMP enabled when CUDA is available (`CONFIG['mixed_precision']`)
- Gradient accumulation: `CONFIG['gradient_accumulation']` to simulate a larger effective batch size
- Gradient clipping: global norm clipped to `CONFIG['gradient_clip']`
- Early stopping/model selection: best checkpoint tracked by validation AUC-ROC; patience `CONFIG['early_stopping_patience']`
- Checkpointing: `checkpoints/best_model.pth` for best; periodic `checkpoints/checkpoint_epoch_*.pth`

The per-epoch flow is:
1) Train one epoch: forward, loss, backward; apply optimizer steps every `gradient_accumulation` batches
2) Validate on the val loader; compute metrics from predictions and probabilities
3) Step LR scheduler after warmup
4) Save best model if AUC-ROC improved; early-stop if no improvement for `patience` epochs

The full training runner in `train_enhanced.py` adds:
- Configuration echo, dataset creation, class-count-aware focal alpha
- Post-training “comprehensive validation” using `src/tester.py`
- Visualization dashboard via `src/visualizer.py` (curves, metrics bar, confusion matrix, ROC, PR) and a JSON/TXT report

## Metrics: definitions and computation

Metrics are implemented in `src/metrics.py` (`MetricsCalculator`) and used in both training validation and comprehensive testing. Predictions are computed from logits via softmax to probabilities and argmax to class labels.

Let the confusion matrix entries be TP (true positives), TN (true negatives), FP (false positives), FN (false negatives). We use the following definitions:

- Accuracy: $\displaystyle \frac{TP + TN}{TP + TN + FP + FN}$
- Precision: $\displaystyle \frac{TP}{TP + FP}$ with zero-division guarded to 0
- Recall (Sensitivity, TPR): $\displaystyle \frac{TP}{TP + FN}$ with zero-division guarded to 0
- Specificity (TNR): $\displaystyle \frac{TN}{TN + FP}$ with zero-division guarded to 0
- F1-score: $\displaystyle 2\cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ with zero-division guarded to 0

Threshold-free metrics computed from positive-class probabilities $p(\text{malignant})$:

- ROC curve: plot of TPR vs FPR, where $\mathrm{FPR} = \frac{FP}{FP + TN}$. AUC-ROC is computed via `sklearn.metrics.roc_auc_score`.
- Precision–Recall curve: precision vs recall across thresholds. AUC-PR (average precision) is computed via `sklearn.metrics.average_precision_score`.

Additional outputs:
- Confusion matrix counts and a seaborn heatmap
- ROC and PR curves saved as images

All computations guard against degenerate cases (e.g., single-class validation targets) by returning 0.0 AUCs and printing a warning.

## Evaluation protocol

During training (`Trainer.validate`):
- After each epoch, compute loss and metrics on the validation loader
- Save the best checkpoint by highest AUC-ROC
- Append to histories for plotting

Comprehensive evaluation (`src/tester.py`):
- Loads the validation split (or any provided CSV) with val transforms
- Computes predictions, probabilities, and metrics
- Saves per-image predictions CSV, error analysis CSV, and a test report JSON
- Creates confusion matrix, ROC, PR, and final-metrics plots under `results/testing/`

## Visualizations and reports

Generated by `src/visualizer.py`:
- Training dashboard (`results/training_..._curves.png`, `results/metrics_..._metrics.png`, `results/confusion_..._matrix.png`, plus ROC/PR when valid)
- Detailed JSON and text reports with configuration, final metrics, and a classification report

## Configuration summary

Primary knobs in `src/config.py` → `CONFIG`:
- Data: `image_size`, `batch_size`, `num_workers`, `mean/std`
- Model: `model_name` (EfficientNet-B3), `pretrained`, `dropout`, `num_classes=2`
- Training: `epochs`, `learning_rate`, `weight_decay`, `warmup_epochs`, `mixed_precision`, `gradient_accumulation`, `gradient_clip`, early stopping
- Loss: `focal_alpha`, `focal_gamma`, optional `label_smoothing`
- Splits/seed: `train_split`, `val_split`, `random_seed`
- Device: auto-detected CUDA vs CPU

Notes:
- `CONFIG` also contains fields for HRNet/feature fusion/WOA/StyleGAN that are not exercised by the current baseline training scripts. They are safe to ignore unless you extend the codebase accordingly.

## Reproducibility

- Random seed: `CONFIG['random_seed']`
- Determinism: full determinism isn’t guaranteed due to data loader parallelism and GPU kernels; for stricter reproducibility, set `num_workers=0` and configure PyTorch/CUDA deterministic flags.
- Environment: Python 3.10+, PyTorch (GPU if available), timm, albumentations, scikit-learn, matplotlib, seaborn, pydicom.

## Limitations and future directions

- Class imbalance remains a challenge; focal loss helps but may benefit from additional strategies (e.g., class-balanced loss, calibrated thresholds).
- Data variability (lighting, artifacts) can affect robustness; consider stain normalization or segmentation-based cropping as future enhancements.
- Advanced architectures and fusion (e.g., HRNet + EfficientNet with metaheuristic fusion, optional GAN augmentation) are mentioned in the configuration/README but aren’t part of the baseline training code in `src/`—they would require additional modules and a dedicated trainer.

## References

- T.-Y. Lin et al., “Focal Loss for Dense Object Detection,” ICCV 2017.
- SIIM-ISIC Melanoma Classification: Kaggle competition page.

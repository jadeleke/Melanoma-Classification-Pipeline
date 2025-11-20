# Melanoma Classification (SIIM-ISIC)

Binary skin lesion classification (benign vs malignant) using **WOA-Hybrid Model** combining HRNet and EfficientNet-B3 with Whale Optimization Algorithm for feature fusion, Albumentations, Focal Loss, and mixed precision training.

## 🆕 Hybrid Model Architecture

This project now implements a **Whale Optimization Algorithm (WOA) based Hybrid Model** that combines:
- **HRNet-W32**: High-resolution spatial feature extraction
- **EfficientNet-B3**: Deep classification features
- **WOA**: Metaheuristic optimization for feature fusion weights
- **StyleGAN2** (optional): Synthetic data augmentation

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

## Highlights

### Baseline Model (EfficientNet-B3)
- EfficientNet-B3 backbone (via timm) with custom classifier head
- Albumentations-heavy training pipeline
- Focal Loss with data-driven alpha from class counts
- Mixed precision (AMP), gradient accumulation, gradient clipping
- Early stopping and best-checkpoint selection by AUC-ROC

### Hybrid Model (NEW)
- **Multi-architecture fusion**: HRNet + EfficientNet
- **WOA optimization**: Automatic feature fusion weight optimization
- **Component-wise learning rates**: Optimized training for each component
- **StyleGAN augmentation** (optional): Synthetic lesion image generation
- **Interpretable fusion**: Track component importance over time
- **Target performance**: >96% AUC-ROC


## Project structure

```
segmentation/
├─ main.py                        # Colab-oriented end-to-end runner (downloads data, etc.)
├─ prepare_data.py                # Stratified train/val split creation
├─ train_enhanced.py              # End-to-end training + comprehensive validation evaluation
├─ requirements.txt               # Dependencies (GPU torch via cu130 find-links)
├─ IMPLEMENTATION.md              # In-depth implementation overview (facts only)
├─ TODO.md                        # Tasks/notes
├─ checkpoints/                   # Model checkpoints
│  └─ best_model.pth             # Best model (AUC-ROC criterion)
├─ data/
│  ├─ train/
│  │  ├─ images/                 # Training images (JPEG or DICOM)
│  │  ├─ train.csv               # Kaggle training metadata
│  │  ├─ train_split.csv         # Generated stratified split
│  │  └─ val_split.csv           # Generated stratified split
│  └─ test/
│     └─ images/                 # Test images (optional)
├─ logs/
│  ├─ training_log_*.json        # Training history snapshots
├─ results/
│  ├─ training_..._curves.png    # Loss & validation metric curves
│  ├─ metrics_..._metrics.png    # Final validation metrics (bar chart)
│  ├─ confusion_..._matrix.png   # Validation confusion matrix
│  ├─ roc_..._curve.png          # Validation ROC curve
│  ├─ pr_..._curve.png           # Validation PR curve
│  ├─ report_..._json/txt        # Validation report (JSON/Text)
│  └─ testing/
│     ├─ test_metrics_...png     # Test/validation-as-test metrics
│     ├─ test_confusion_...png   # Test/validation-as-test confusion matrix
│     ├─ test_roc_...png         # Test ROC curve
│     ├─ test_pr_...png          # Test PR curve
│     ├─ test_predictions_...csv # Per-image predictions
│     ├─ error_analysis_...csv   # Misclassifications
│     └─ test_report_...json     # Test report
└─ src/
   ├─ config.py                  # Hyperparameters, device, paths
   ├─ dataset.py                 # Dataset, DICOM/JPEG loading, transforms, dataloaders
   ├─ loss.py                    # Focal Loss and LabelSmoothingCE
   ├─ metrics.py                 # Metrics calculator and simple plots
   ├─ model.py                   # EfficientNet-B3 + classifier head
   ├─ train.py                   # Trainer: AMP, accumulation, clipping, early stop, sched, ckpt
   ├─ tester.py                  # Comprehensive evaluation & reports
   └─ visualizer.py             # Dashboards and plots
```


## Installation (cross‑platform)

Requires Python 3.10+.

1) Create a virtual environment

- Windows (PowerShell):

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

- Windows (cmd):

```bat
python -m venv .venv
.\.venv\Scripts\activate.bat
```

- macOS/Linux (bash/zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies

```bash
pip install -r requirements.txt
```

Notes:
- `requirements.txt` includes `--find-links https://download.pytorch.org/whl/cu130` so pip resolves GPU-enabled wheels for CUDA 13.0.
- Ensure your NVIDIA driver supports CUDA 13.0; otherwise, adjust the find-links URL for your CUDA version.
- `pydicom` is included for DICOM support.


## Data: download and prepare (browser)

This project uses the Kaggle SIIM-ISIC Melanoma Classification dataset. You can download it via the browser (no CLI required):

1) Open the competition page in your browser
- https://www.kaggle.com/competitions/siim-isic-melanoma-classification
- Sign in and accept the competition rules.

2) Download archives
- Download at minimum:
  - The `jpeg` directory.
  - You can also download the other file formats

3) Create the data folder
- Create a folder named `data` in the project root.

4) Extract and organize
- Extract the downloaded archives into `data/`.
- Ensure the final structure contains:
  - `data/train/images/` (JPEG files)
  - `data/test/images/` (JPEG files, optional)
  - `data/train/train.csv`
- If your download extracts into a `jpeg/` tree, move files accordingly:
  - Move `data/jpeg/train/*.jpg` → `data/train/images/`
  - Move `data/jpeg/test/*.jpg`  → `data/test/images/`

5) Create stratified train/val splits

```bash
python prepare_data.py
```

Outputs:
- `data/train/train_split.csv`
- `data/train/val_split.csv`


## Training

### Hybrid Model (Recommended)

Run the hybrid model training with WOA optimization:

```bash
python train_hybrid.py
```

What it does:
- Loads hybrid configuration from `src/config.py`
- Builds WOA-Hybrid model (HRNet + EfficientNet)
- Creates dataloaders with optional StyleGAN augmentation
- Trains with:
  - Dual-phase: warmup → WOA optimization
  - Component-wise learning rates
  - Mixed precision, gradient accumulation
  - Periodic WOA optimization (every 5 epochs)
  - Early stopping based on validation AUC
- Saves best checkpoint to `checkpoints/best_model.pth`
- Logs fusion weights and WOA convergence

### Baseline Model (Original)

Run the original EfficientNet-B3 pipeline:

```bash
python train_enhanced.py
```

What it does:
- Trains EfficientNet-B3 only
- Same training optimizations
- Faster training (~4 hours vs 6-10 hours)
- Lower memory usage
- Good baseline for comparison


## Configuration

### Hybrid Model Configuration

Key CONFIG fields in `src/config.py`:

**Model Selection**:
- `model_type`: `'hybrid'` (or `'baseline'` for EfficientNet-only)

**HRNet**:
- `hrnet.model_name`: `'hrnet_w32'` (or `'hrnet_w48'`)
- `hrnet.feature_dim`: `2048`
- `hrnet.freeze_backbone`: `False`

**EfficientNet**:
- `efficientnet.model_name`: `'efficientnet_b3'`
- `efficientnet.feature_dim`: `1536`
- `efficientnet.dropout`: `0.3`

**Fusion**:
- `fusion.strategy`: `'weighted_sum'` (or `'concat'`)
- `fusion.initial_weights`: `[0.5, 0.5]`
- `fusion.learnable`: `True`

**WOA Optimization**:
- `woa.enabled`: `True`
- `woa.n_whales`: `20` (population size)
- `woa.max_iter`: `30` (iterations per optimization)
- `woa.optimization_frequency`: `5` (optimize every N epochs)
- `woa.start_epoch`: `10` (start after warmup)

**StyleGAN** (optional):
- `stylegan.enabled`: `False`
- `stylegan.synthetic_ratio`: `0.2` (20% synthetic images)
- `stylegan.generator_path`: `None` (path to pretrained generator)

**Training**:
- Data: `image_size=256`, `batch_size=8`, `num_workers=4`
- Epochs: `30`, Learning rate: `1e-4`, Weight decay: `1e-2`
- Loss: Focal Loss (`alpha=0.25`, `gamma=2.0`)
- Optimization: `gradient_accumulation=4`, `mixed_precision=True`

### Baseline Model Configuration

Same as above, but set:
- `model_type`: `'baseline'`
- All other settings remain compatible


## Model & training details

- Backbone: EfficientNet-B3 (timm), features → `Dropout(0.3)` → `Linear(num_features, 2)`
- Transforms (train): resize → flips (H/V) → rotate (90°) → shift/scale/rotate → color jitter (brightness/contrast, HSV, gamma) → blur → coarse dropout → normalize → tensor
- Transforms (val/test): resize → normalize → tensor
- Loss: Focal Loss (per-sample CE · focal factor), alpha derived from class counts when available
- Metrics: Accuracy, Precision, Recall (Sensitivity), Specificity, F1, AUC-ROC, AUC-PR
- Scheduler: CosineAnnealingWarmRestarts; stepped after warmup epochs
- Early stopping / model selection metric: AUC-ROC


## Results (artifacts)

Latest run artifacts in this workspace:

### Validation (during training)

Below are direct embeds of the latest validation artifacts captured in this workspace:

![Training & Validation Curves](results/training_20251102_234409_curves.png)

![Final Validation Metrics](results/metrics_20251102_234409_metrics.png)

![Validation Confusion Matrix](results/confusion_20251102_234409_matrix.png)

![Validation ROC Curve](results/roc_20251102_234409_curve.png)

![Validation PR Curve](results/pr_20251102_234409_curve.png)

Additional reports:
- JSON: [results/report_20251102_234409.json](results/report_20251102_234409.json)
- Text: [results/report_20251102_234409.txt](results/report_20251102_234409.txt)

### Comprehensive evaluation (validation-as-test)

![Test Metrics (Validation-as-Test)](results/testing/test_metrics_20251102_234705_metrics.png)

![Test Confusion Matrix](results/testing/test_confusion_20251102_234705_matrix.png)

![Test ROC Curve](results/testing/test_roc_20251102_234705_curve.png)

![Test PR Curve](results/testing/test_pr_20251102_234705_curve.png)

Data tables:
- Predictions: [results/testing/test_predictions_20251102_234705.csv](results/testing/test_predictions_20251102_234705.csv)
- Error analysis: [results/testing/error_analysis_20251102_234705.csv](results/testing/error_analysis_20251102_234705.csv)
- Report: [results/testing/test_report_20251102_234705.json](results/testing/test_report_20251102_234705.json)


## Checkpoints and logs

- Best checkpoint: `checkpoints/best_model.pth` (contains model/optimizer/scheduler/scaler states and histories)
- Periodic checkpoints: `checkpoints/checkpoint_epoch_*.pth` every 5 epochs
- Logs: `logs/training_log_*.json` (config, losses, metrics per epoch)


## How to evaluate a trained model on a CSV

`train_enhanced.py` performs evaluation on the validation split automatically. To evaluate another split/test set, use `src/tester.py` programmatically in a small script:

```python
from src.model import create_model
from src.config import CONFIG
from src.tester import ModelTester

model = create_model(CONFIG)
# load weights if needed
# model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=CONFIG['device'])['model_state_dict'])

tester = ModelTester(model, CONFIG, CONFIG['device'])
metrics, df = tester.run_comprehensive_test('path/to/your.csv', 'path/to/images')
print(metrics)
```


## Troubleshooting

- Data not found: ensure `data/train/images/` and `data/train/train.csv` exist before running `prepare_data.py` and training.
- CSV schema: must include `image_name` and `target` (0/1) columns.
- DICOM images: supported via `pydicom` (`.dcm`), auto-detected in `src/dataset.py`.
- GPU build: `requirements.txt` points to cu130 wheels; ensure your NVIDIA driver/toolkit supports CUDA 13.0.
- Colab runner: `main.py` calls `train_model.py`, which is not present in this repo; use `train_enhanced.py` locally.


## References

- Implementation details: see [`IMPLEMENTATION.md`](IMPLEMENTATION.md)
- Dataset: SIIM-ISIC Melanoma Classification (Kaggle)
#   M e l a n o m a - C l a s s i f i c a t i o n - P i p e l i n e  
 
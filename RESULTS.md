# Results Report

This report summarizes the results produced by the training and evaluation runs recorded in this workspace. All values and artifacts referenced here are taken directly from the generated JSON reports, logs, and images under `results/`, `results/testing/`, and `logs/`.

## Run summary

- Training time (reported): 4.3826 hours (`results/report_20251102_234409.json`)
- Epochs completed: 19 (`logs/training_log_20251102_234409.json`)
- Best validation AUC-ROC (during training): 0.9216 at epoch 9 (`logs/training_log_20251102_234409.json`)
- Final validation metrics snapshot (end of recorded training history): see “Validation results” below
- Comprehensive evaluation on the validation split used as a test set: see “Comprehensive evaluation (validation-as-test)”

## Configuration snapshot

From `results/report_20251102_234409.json` and `logs/training_log_20251102_234409.json`:
- Model: EfficientNet-B3 (pretrained=True), Dropout=0.3, num_classes=2
- Image size: 256, Batch size: 8, Num workers: 4
- Optimizer: AdamW, lr=1e-4, weight_decay=1e-2
- Scheduler: CosineAnnealingWarmRestarts with warmup (5 epochs)
- Loss: Focal Loss (alpha=0.25, gamma=2.0), label_smoothing=0.1 (not applied inside focal loss)
- AMP: enabled, Gradient accumulation: 4, Gradient clip: 1.0
- Early stopping patience: 10
- Device: cuda

## Validation results (training-time)

Source: `results/report_20251102_234409.json`

- Accuracy: 0.9419
- Precision: 0.1474
- Recall (Sensitivity): 0.4786
- Specificity: 0.9502
- F1-score: 0.2254
- AUC-ROC: 0.9062
- AUC-PR: 0.1776
- Confusion matrix (rows: true [0,1], cols: pred [0,1]):
  - TN=6185, FP=324, FN=61, TP=56
- Class supports (from classification report):
  - Class 0 support: 6509
  - Class 1 support: 117

Validation artifacts:

- Training & validation curves: `results/training_20251102_234409_curves.png`

  ![Training & Validation Curves](results/training_20251102_234409_curves.png)

- Final validation metrics: `results/metrics_20251102_234409_metrics.png`

  ![Final Validation Metrics](results/metrics_20251102_234409_metrics.png)

- Confusion matrix: `results/confusion_20251102_234409_matrix.png`

  ![Validation Confusion Matrix](results/confusion_20251102_234409_matrix.png)

- ROC curve: `results/roc_20251102_234409_curve.png`

  ![Validation ROC Curve](results/roc_20251102_234409_curve.png)

- PR curve: `results/pr_20251102_234409_curve.png`

  ![Validation PR Curve](results/pr_20251102_234409_curve.png)

- Reports:
  - JSON: [results/report_20251102_234409.json](results/report_20251102_234409.json)
  - Text: [results/report_20251102_234409.txt](results/report_20251102_234409.txt)

### Classification report (validation)

Extracted from `results/report_20251102_234409.json`:

- Class 0:
  - precision: 0.9902, recall: 0.9502, f1-score: 0.9698, support: 6509
- Class 1:
  - precision: 0.1474, recall: 0.4786, f1-score: 0.2254, support: 117
- Accuracy: 0.9419
- Macro avg (precision/recall/f1): 0.5688 / 0.7144 / 0.5976
- Weighted avg (precision/recall/f1): 0.9754 / 0.9419 / 0.9567

## Comprehensive evaluation (validation-as-test)

Source: `results/testing/test_report_20251102_234705.json`

- Accuracy: 0.9054
- Precision: 0.1205
- Recall (Sensitivity): 0.6923
- Specificity: 0.9092
- F1-score: 0.2053
- AUC-ROC: 0.9215
- AUC-PR: 0.2292
- Test loss: 0.3108
- Confusion matrix (rows: true [0,1], cols: pred [0,1]):
  - TN=5918, FP=591, FN=36, TP=81
- Sample counts:
  - total: 6626
  - benign: 6509
  - malignant: 117
  - correct_predictions: 5999
  - false_positives: 591
  - false_negatives: 36

Evaluation artifacts:

- Metrics (bar): `results/testing/test_metrics_20251102_234705_metrics.png`

  ![Test Metrics (Validation-as-Test)](results/testing/test_metrics_20251102_234705_metrics.png)

- Confusion matrix: `results/testing/test_confusion_20251102_234705_matrix.png`

  ![Test Confusion Matrix](results/testing/test_confusion_20251102_234705_matrix.png)

- ROC curve: `results/testing/test_roc_20251102_234705_curve.png`

  ![Test ROC Curve](results/testing/test_roc_20251102_234705_curve.png)

- PR curve: `results/testing/test_pr_20251102_234705_curve.png`

  ![Test PR Curve](results/testing/test_pr_20251102_234705_curve.png)

- Data tables:
  - Predictions: [results/testing/test_predictions_20251102_234705.csv](results/testing/test_predictions_20251102_234705.csv)
  - Error analysis: [results/testing/error_analysis_20251102_234705.csv](results/testing/error_analysis_20251102_234705.csv)
  - Report: [results/testing/test_report_20251102_234705.json](results/testing/test_report_20251102_234705.json)

## Training dynamics

From `logs/training_log_20251102_234409.json`:

- Epochs completed: 19
- Best AUC-ROC during training: 0.9216 (epoch 9)
- Final recorded losses:
  - Train loss (epoch 19): 0.001630
  - Val loss (epoch 19): 0.009119
- Validation metric trajectory included swings in sensitivity/specificity trade-offs while maintaining AUC-ROC in the ~0.86–0.92 range across epochs.

## Artifacts index

- Checkpoint:
  - `checkpoints/best_model.pth`
- Logs:
  - `logs/training_log_20251102_234409.json`
- Validation (training-time) images:
  - `results/training_20251102_234409_curves.png`
  - `results/metrics_20251102_234409_metrics.png`
  - `results/confusion_20251102_234409_matrix.png`
  - `results/roc_20251102_234409_curve.png`
  - `results/pr_20251102_234409_curve.png`
- Validation (training-time) reports:
  - [results/report_20251102_234409.json](results/report_20251102_234409.json)
  - [results/report_20251102_234409.txt](results/report_20251102_234409.txt)
- Comprehensive evaluation (validation-as-test) images:
  - `results/testing/test_metrics_20251102_234705_metrics.png`
  - `results/testing/test_confusion_20251102_234705_matrix.png`
  - `results/testing/test_roc_20251102_234705_curve.png`
  - `results/testing/test_pr_20251102_234705_curve.png`
- Comprehensive evaluation (validation-as-test) tables/reports:
  - [results/testing/test_predictions_20251102_234705.csv](results/testing/test_predictions_20251102_234705.csv)
  - [results/testing/error_analysis_20251102_234705.csv](results/testing/error_analysis_20251102_234705.csv)
  - [results/testing/test_report_20251102_234705.json](results/testing/test_report_20251102_234705.json)

## Reproduction

- Environment: see `requirements.txt` (GPU-enabled torch with CUDA 13.0 via cu130 wheel index)
- Data splits: created by `prepare_data.py`
- Training/Evaluation: `train_enhanced.py` (trains then evaluates on validation-as-test)

All numbers and paths in this document are taken directly from the files produced during the referenced run(s).
"""
Test dataset evaluation after training.
Provides comprehensive testing and analysis on the test set.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from src.dataset import MelanomaDataset
from src.metrics import MetricsCalculator
from src.visualizer import TrainingVisualizer


class ModelTester:
    """
    Comprehensive testing suite for trained models.
    """
    
    def __init__(self, model, config, device='cuda'):
        self.model = model
        self.config = config
        self.device = device
        self.model.eval()
        
        # Results directory
        self.results_dir = Path("results/testing")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_test_data(self, test_csv_path, test_img_dir):
        """Load test dataset."""
        print(f"Loading test data from: {test_csv_path}")
        
        # Import transforms
        from src.dataset import get_val_transforms
        
        # Create test dataset with proper transforms
        test_dataset = MelanomaDataset(
            csv_path=test_csv_path,
            img_dir=test_img_dir,
            image_size=self.config['image_size'],
            transform=get_val_transforms(
                image_size=self.config['image_size'],
                mean=self.config['mean'],
                std=self.config['std']
            ),
            mode='val'  # Use validation mode (no augmentation)
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        print(f"Test dataset: {len(test_dataset)} samples")
        return test_loader
    
    @torch.no_grad()
    def evaluate_model(self, test_loader):
        """Evaluate model on test set."""
        print(f"\\n{'='*60}")
        print("EVALUATING MODEL ON TEST SET")
        print(f"{'='*60}")
        
        self.model.eval()
        
        all_predictions = []
        all_probabilities = []
        all_targets = []
        all_image_names = []
        
        total_loss = 0.0
        num_batches = 0
        
        # Create loss function for evaluation
        criterion = nn.CrossEntropyLoss()
        
        pbar = tqdm(test_loader, desc="Testing")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            targets = batch['target'].to(self.device)
            image_names = batch['image_name']
            
            # Forward pass
            outputs = self.model(images)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # Store results
            all_predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
            all_targets.extend(targets.cpu().numpy())
            all_image_names.extend(image_names)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss/num_batches:.4f}"
            })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        return {
            'predictions': np.array(all_predictions),
            'probabilities': np.array(all_probabilities),
            'targets': np.array(all_targets),
            'image_names': all_image_names,
            'test_loss': avg_loss
        }
    
    def calculate_detailed_metrics(self, results):
        """Calculate comprehensive metrics on test results."""
        print(f"\\nCalculating detailed metrics...")
        
        # Use MetricsCalculator
        metrics_calc = MetricsCalculator()
        
        # Convert back to torch tensors with proper shape for metrics calculator
        preds_tensor = torch.tensor(results['predictions'])
        probs_2d = torch.zeros(len(results['probabilities']), 2)
        probs_2d[:, 0] = torch.tensor(1 - results['probabilities'])  # Benign probability
        probs_2d[:, 1] = torch.tensor(results['probabilities'])      # Malignant probability
        targets_tensor = torch.tensor(results['targets'])
        
        metrics_calc.update(preds_tensor, probs_2d, targets_tensor)
        detailed_metrics = metrics_calc.compute()
        
        # Add test loss
        detailed_metrics['test_loss'] = results['test_loss']
        
        return detailed_metrics
    
    def save_predictions(self, results, save_prefix="test_predictions"):
        """Save detailed predictions for analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'image_name': results['image_names'],
            'true_label': results['targets'],
            'predicted_label': results['predictions'],
            'malignant_probability': results['probabilities'],
            'benign_probability': 1 - results['probabilities'],
            'correct_prediction': results['targets'] == results['predictions']
        })
        
        # Add confidence levels
        predictions_df['confidence'] = np.abs(results['probabilities'] - 0.5) * 2
        predictions_df['confidence_level'] = pd.cut(
            predictions_df['confidence'], 
            bins=[0, 0.2, 0.6, 1.0], 
            labels=['Low', 'Medium', 'High']
        )
        
        # Save to CSV
        csv_path = self.results_dir / f"{save_prefix}_{timestamp}.csv"
        predictions_df.to_csv(csv_path, index=False)
        
        print(f"Predictions saved to: {csv_path}")
        
        # Print summary statistics
        print(f"\\nPrediction Summary:")
        print(f"Total samples: {len(predictions_df)}")
        print(f"Correct predictions: {predictions_df['correct_prediction'].sum()}")
        print(f"Accuracy: {predictions_df['correct_prediction'].mean():.4f}")
        print(f"\\nConfidence distribution:")
        print(predictions_df['confidence_level'].value_counts())
        
        return predictions_df
    
    def analyze_errors(self, results, predictions_df, save_prefix="error_analysis"):
        """Analyze incorrect predictions."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get incorrect predictions
        errors = predictions_df[~predictions_df['correct_prediction']].copy()
        
        if len(errors) == 0:
            print("\\nNo prediction errors found!")
            return
        
        print(f"\\n{'='*50}")
        print(f"ERROR ANALYSIS")
        print(f"{'='*50}")
        print(f"Total errors: {len(errors)} / {len(predictions_df)} ({len(errors)/len(predictions_df)*100:.1f}%)")
        
        # Analyze error types
        false_positives = errors[errors['predicted_label'] == 1]  # Predicted malignant, actually benign
        false_negatives = errors[errors['predicted_label'] == 0]  # Predicted benign, actually malignant
        
        print(f"False Positives (predicted malignant, actually benign): {len(false_positives)}")
        print(f"False Negatives (predicted benign, actually malignant): {len(false_negatives)}")
        
        # Save error analysis
        error_path = self.results_dir / f"{save_prefix}_{timestamp}.csv"
        errors.to_csv(error_path, index=False)
        print(f"\\nError analysis saved to: {error_path}")
        
        # Print most confident errors
        if len(errors) > 0:
            print(f"\\nMost confident errors (top 10):")
            top_errors = errors.nlargest(min(10, len(errors)), 'confidence')
            for _, row in top_errors.iterrows():
                print(f"  {row['image_name']}: True={row['true_label']}, Pred={row['predicted_label']}, "
                      f"Conf={row['confidence']:.3f}, Prob(Mal)={row['malignant_probability']:.3f}")
    
    def run_comprehensive_test(self, test_csv_path, test_img_dir):
        """Run complete testing pipeline."""
        print(f"\\n{'='*80}")
        print("COMPREHENSIVE MODEL TESTING")
        print(f"{'='*80}")
        
        start_time = datetime.now()
        
        # Load test data
        test_loader = self.load_test_data(test_csv_path, test_img_dir)
        
        # Evaluate model
        results = self.evaluate_model(test_loader)
        
        # Calculate metrics
        detailed_metrics = self.calculate_detailed_metrics(results)
        
        # Print metrics
        print(f"\\n{'='*60}")
        print("TEST RESULTS")
        print(f"{'='*60}")
        print(f"Test Loss: {detailed_metrics['test_loss']:.4f}")
        print(f"Accuracy: {detailed_metrics['accuracy']:.4f}")
        print(f"Precision: {detailed_metrics['precision']:.4f}")
        print(f"Recall (Sensitivity): {detailed_metrics['recall']:.4f}")
        print(f"Specificity: {detailed_metrics['specificity']:.4f}")
        print(f"F1-Score: {detailed_metrics['f1']:.4f}")
        print(f"AUC-ROC: {detailed_metrics['auc_roc']:.4f}")
        print(f"AUC-PR: {detailed_metrics['auc_pr']:.4f}")
        print(f"\\nConfusion Matrix:")
        print(f"  TN: {detailed_metrics['tn']:5d}  |  FP: {detailed_metrics['fp']:5d}")
        print(f"  FN: {detailed_metrics['fn']:5d}  |  TP: {detailed_metrics['tp']:5d}")
        
        # Save predictions
        predictions_df = self.save_predictions(results)
        
        # Error analysis
        self.analyze_errors(results, predictions_df)
        
        # Create visualizations
        print(f"\\nCreating test visualizations...")
        visualizer = TrainingVisualizer(self.results_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create plots
        visualizer.plot_confusion_matrix(results['targets'], results['predictions'], f"test_confusion_{timestamp}")
        visualizer.plot_roc_curves(results['targets'], results['probabilities'], f"test_roc_{timestamp}")
        visualizer.plot_precision_recall_curve(results['targets'], results['probabilities'], f"test_pr_{timestamp}")
        visualizer.plot_final_metrics(detailed_metrics, f"test_metrics_{timestamp}")
        
        # Save test report
        test_report = {
            'timestamp': start_time.isoformat(),
            'test_duration_minutes': (datetime.now() - start_time).total_seconds() / 60,
            'model_config': self.config,
            'test_metrics': detailed_metrics,
            'sample_counts': {
                'total_samples': len(results['targets']),
                'benign_samples': int(np.sum(results['targets'] == 0)),
                'malignant_samples': int(np.sum(results['targets'] == 1)),
                'correct_predictions': int(np.sum(results['targets'] == results['predictions'])),
                'false_positives': int(detailed_metrics['fp']),
                'false_negatives': int(detailed_metrics['fn'])
            }
        }
        
        # Save report
        import json
        report_path = self.results_dir / f"test_report_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(test_report, f, indent=2, default=str)
        
        print(f"\\n{'='*80}")
        print(f"TESTING COMPLETE!")
        print(f"Duration: {test_report['test_duration_minutes']:.1f} minutes")
        print(f"Results saved to: {self.results_dir}")
        print(f"{'='*80}\\n")
        
        return detailed_metrics, predictions_df


if __name__ == "__main__":
    print("Model testing module ready")
    print("Use after training: tester = ModelTester(model, config); tester.run_comprehensive_test(...)")
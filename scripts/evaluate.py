#!/usr/bin/env python3
"""
Evaluation script for trained R-peak detection models.
"""

import sys
import os
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.utils.config import load_config, get_device_config
from src.utils.helpers import set_seed, print_system_info, Timer
from src.data.loader import load_all_files
from src.preprocessing.signal_processing import prepare_data_splits
from src.data.dataset import create_datasets
from src.models import get_model
from src.training.trainer import RPeakTrainer
from src.training.metrics import (
    evaluate_at_multiple_thresholds,
    find_optimal_threshold,
    print_metrics
)
from src.visualization.plots import plot_prediction_results, plot_confusion_matrix


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate trained R-peak detection model')

    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold')
    parser.add_argument('--find-optimal-threshold', action='store_true',
                       help='Find optimal threshold based on F1 score')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu, cuda, auto)')
    parser.add_argument('--output-dir', type=str, default='outputs/evaluation',
                       help='Output directory for evaluation results')
    parser.add_argument('--create-plots', action='store_true',
                       help='Create visualization plots')

    return parser.parse_args()


def load_trained_model(model_path: str, config, device: torch.device):
    """Load trained model from checkpoint."""
    print(f"Loading model from: {model_path}")

    # Create model
    model = get_model(
        model_name=config.model.name,
        input_size=config.model.input_size,
        dropout_rate=config.model.dropout_rate,
        num_blocks=getattr(config.model, 'num_blocks', 4),
        base_channels=getattr(config.model, 'base_channels', 32)
    )

    # Create trainer and load model
    trainer = RPeakTrainer(model, device)
    trainer.load_model(model_path)

    print(f"Model loaded successfully: {model.__class__.__name__}")
    print(f"Parameters: {model.get_num_parameters():,}")

    return trainer


def evaluate_model_comprehensive(trainer: RPeakTrainer, test_loader: DataLoader,
                                thresholds: list = None):
    """Comprehensive model evaluation."""
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    print("Running comprehensive evaluation...")

    # Get predictions and scores
    trainer.model.eval()
    all_predictions = []
    all_targets = []
    all_scores = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(trainer.device)
            output = trainer.model(data)

            # Handle different output types
            if hasattr(trainer.criterion, '__class__') and 'Logits' in str(trainer.criterion.__class__):
                scores = torch.sigmoid(output)
            else:
                scores = output

            all_scores.extend(scores.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy().flatten())

    all_scores = np.array(all_scores)
    all_targets = np.array(all_targets)

    # Evaluate at multiple thresholds
    threshold_results = evaluate_at_multiple_thresholds(all_targets, all_scores, thresholds)

    return all_targets, all_scores, threshold_results


def main():
    """Main evaluation pipeline."""
    args = parse_arguments()

    # Load configuration
    config = load_config(args.config)

    # Setup
    set_seed(config.seed)
    device = get_device_config(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print_system_info()

    try:
        # Load trained model
        trainer = load_trained_model(args.model_path, config, device)

        # Load and prepare data
        print("Loading test data...")
        with Timer("Data loading"):
            all_eeg, all_rpeaks, all_ecg, all_rpeak_locations = load_all_files(
                file_pattern=config.data.file_pattern,
                max_samples=config.data.max_samples,
                eeg_lowcut=config.data.eeg_lowcut,
                eeg_highcut=config.data.eeg_highcut,
                sampling_rate=config.data.sampling_rate
            )

        # Prepare data splits (we only need test set)
        with Timer("Data preparation"):
            _, _, eeg_test, _, _, rpeaks_test, _ = prepare_data_splits(
                eeg_data=all_eeg,
                rpeak_data=all_rpeaks,
                train_ratio=config.data.train_ratio,
                val_ratio=config.data.val_ratio
            )

        # Create test dataset and loader
        _, _, test_dataset = create_datasets(
            eeg_train=np.array([]),  # Empty arrays for train/val
            eeg_val=np.array([]),
            eeg_test=eeg_test,
            rpeaks_train=np.array([]),
            rpeaks_val=np.array([]),
            rpeaks_test=rpeaks_test,
            window_size=config.data.window_size,
            balanced=False  # No need for balancing in evaluation
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=2
        )

        print(f"Test set: {len(test_dataset)} samples")

        # Run comprehensive evaluation
        with Timer("Model evaluation"):
            all_targets, all_scores, threshold_results = evaluate_model_comprehensive(trainer, test_loader)

        # Print results for different thresholds
        print("\n=== Evaluation Results at Different Thresholds ===")
        for threshold, metrics in threshold_results.items():
            print(f"\n--- Threshold {threshold:.1f} ---")
            print_metrics(metrics, f"Threshold {threshold:.1f}")

        # Find optimal threshold if requested
        if args.find_optimal_threshold:
            optimal_threshold, best_f1 = find_optimal_threshold(all_targets, all_scores, 'f1_score')
            print(f"\nOptimal threshold: {optimal_threshold:.3f} (F1 = {best_f1:.4f})")

            # Evaluate at optimal threshold
            optimal_predictions = (all_scores > optimal_threshold).astype(int)
            from src.training.metrics import calculate_classification_metrics
            optimal_metrics = calculate_classification_metrics(all_targets, optimal_predictions, all_scores)
            print_metrics(optimal_metrics, f"Optimal Threshold ({optimal_threshold:.3f})")
        else:
            optimal_threshold = args.threshold
            optimal_predictions = (all_scores > optimal_threshold).astype(int)

        # Create plots if requested
        if args.create_plots:
            print("\nCreating visualization plots...")

            # Confusion matrix
            plot_confusion_matrix(
                all_targets, optimal_predictions,
                save_path=os.path.join(args.output_dir, "confusion_matrix.png")
            )

            # Prediction results
            plot_prediction_results(
                eeg_signal=eeg_test,
                true_rpeaks=rpeaks_test,
                predicted_rpeaks=optimal_predictions,
                prediction_scores=all_scores,
                sampling_rate=config.data.sampling_rate,
                duration=20.0,
                save_path=os.path.join(args.output_dir, "prediction_results.png")
            )

        # Save evaluation results
        results = {
            'model_path': args.model_path,
            'threshold_results': threshold_results,
            'test_samples': len(all_targets),
            'positive_samples': int(np.sum(all_targets)),
            'negative_samples': int(len(all_targets) - np.sum(all_targets))
        }

        if args.find_optimal_threshold:
            results['optimal_threshold'] = optimal_threshold
            results['optimal_metrics'] = optimal_metrics

        # Save results
        import json
        results_path = os.path.join(args.output_dir, "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=float)

        print(f"\n=== Evaluation Completed ===")
        print(f"Results saved to: {args.output_dir}")

    except Exception as e:
        print(f"Evaluation failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
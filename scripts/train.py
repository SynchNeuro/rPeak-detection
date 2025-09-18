#!/usr/bin/env python3
"""
Main training script for R-peak detection using EEG signals.

This script provides a unified interface to train different models using
the modular components.
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.utils.config import (
    load_config, save_config, create_default_configs,
    get_device_config, validate_config, print_config, RPeakConfig
)
from src.utils.helpers import (
    set_seed, create_output_directories, save_results,
    print_system_info, Timer, check_data_quality, print_data_quality_report
)
from src.data.loader import load_all_files, get_data_statistics
from src.preprocessing.signal_processing import prepare_data_splits
from src.data.dataset import create_datasets
from src.models import get_model
from src.training.trainer import RPeakTrainer
from src.visualization.plots import create_ecg_validation_plot, plot_training_curves


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train R-peak detection model from EEG signals')

    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, default=None,
                       help='Model name (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cpu, cuda, auto)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides config)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (overrides config)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable plot generation')
    parser.add_argument('--create-default-configs', action='store_true',
                       help='Create default configuration files and exit')
    parser.add_argument('--dry-run', action='store_true',
                       help='Load data and validate config without training')

    return parser.parse_args()


def apply_argument_overrides(config: RPeakConfig, args):
    """Apply command line argument overrides to config."""
    if args.model:
        config.model.name = args.model
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.device:
        config.device = args.device
    if args.seed:
        config.seed = args.seed
    if args.output_dir:
        config.output.base_dir = args.output_dir
        config.output.models_dir = os.path.join(args.output_dir, "models")
        config.output.plots_dir = os.path.join(args.output_dir, "plots")
        config.output.logs_dir = os.path.join(args.output_dir, "logs")
    if args.no_plots:
        config.visualization.create_plots = False


def load_and_prepare_data(config: RPeakConfig):
    """Load and prepare data for training."""
    print("=== Data Loading and Preparation ===")

    with Timer("Data loading"):
        # Load all files
        all_eeg, all_rpeaks, all_ecg, all_rpeak_locations = load_all_files(
            file_pattern=config.data.file_pattern,
            max_samples=config.data.max_samples,
            eeg_lowcut=config.data.eeg_lowcut,
            eeg_highcut=config.data.eeg_highcut,
            sampling_rate=config.data.sampling_rate
        )

    # Data quality check
    quality_report = check_data_quality(all_eeg, all_rpeaks)
    print_data_quality_report(quality_report)

    # Get data statistics
    stats = get_data_statistics(all_eeg, all_rpeaks)
    print("=== Dataset Statistics ===")
    print(f"Total files: {stats['total_files']}")
    print(f"Total samples: {stats['total_samples']:,}")
    print(f"Total R-peaks: {stats['total_rpeaks']:,}")
    print(f"R-peak percentage: {stats['rpeak_percentage']:.3f}%")
    print(f"Class imbalance ratio: 1:{stats['imbalance_ratio']:.1f}")
    print()

    # Create ECG validation plot if enabled
    if config.visualization.create_plots and len(all_ecg) > 0:
        with Timer("ECG validation plot creation"):
            heart_rate = create_ecg_validation_plot(
                ecg=all_ecg[0],
                rpeak_binary=all_rpeaks[0],
                rpeak_locations=all_rpeak_locations[0],
                eeg_filtered=all_eeg[0],
                sampling_rate=config.data.sampling_rate,
                duration=config.visualization.ecg_validation_duration,
                save_path=os.path.join(config.output.plots_dir, "ecg_validation.png")
            )
            if heart_rate:
                print(f"Detected heart rate: {heart_rate:.1f} BPM")

    # Prepare data splits
    with Timer("Data preparation"):
        eeg_train, eeg_val, eeg_test, rpeaks_train, rpeaks_val, rpeaks_test, scaler = prepare_data_splits(
            eeg_data=all_eeg,
            rpeak_data=all_rpeaks,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio
        )

    print(f"Train: {len(eeg_train):,} samples, R-peaks: {np.sum(rpeaks_train):,} ({np.sum(rpeaks_train)/len(rpeaks_train)*100:.3f}%)")
    print(f"Val:   {len(eeg_val):,} samples, R-peaks: {np.sum(rpeaks_val):,} ({np.sum(rpeaks_val)/len(rpeaks_val)*100:.3f}%)")
    print(f"Test:  {len(eeg_test):,} samples, R-peaks: {np.sum(rpeaks_test):,} ({np.sum(rpeaks_test)/len(rpeaks_test)*100:.3f}%)")
    print()

    return eeg_train, eeg_val, eeg_test, rpeaks_train, rpeaks_val, rpeaks_test, scaler, stats


def create_data_loaders(config: RPeakConfig, eeg_train, eeg_val, eeg_test,
                       rpeaks_train, rpeaks_val, rpeaks_test):
    """Create PyTorch data loaders."""
    print("=== Creating Data Loaders ===")

    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(
        eeg_train=eeg_train,
        eeg_val=eeg_val,
        eeg_test=eeg_test,
        rpeaks_train=rpeaks_train,
        rpeaks_val=rpeaks_val,
        rpeaks_test=rpeaks_test,
        window_size=config.data.window_size,
        balanced=True  # Use balanced dataset for better handling of class imbalance
    )

    # Print dataset information
    if hasattr(train_dataset, 'get_class_distribution'):
        class_dist = train_dataset.get_class_distribution()
        print(f"Training class distribution:")
        print(f"  Positive samples: {class_dist['positive_samples']:,} ({class_dist['positive_ratio']*100:.3f}%)")
        print(f"  Negative samples: {class_dist['negative_samples']:,}")
        print(f"  Imbalance ratio: 1:{class_dist['imbalance_ratio']:.1f}")
        print(f"  Class weights: Neg={class_dist['class_weights']['negative']:.4f}, Pos={class_dist['class_weights']['positive']:.4f}")

    # Create data loaders
    train_sampler = None
    if config.training.use_balanced_sampler and hasattr(train_dataset, 'create_balanced_sampler'):
        train_sampler = train_dataset.create_balanced_sampler()
        print("Using balanced sampler for training")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")
    print()

    return train_loader, val_loader, test_loader


def setup_training(config: RPeakConfig, device: torch.device):
    """Setup model and trainer."""
    print("=== Model and Trainer Setup ===")

    # Create model
    model = get_model(
        model_name=config.model.name,
        input_size=config.model.input_size,
        dropout_rate=config.model.dropout_rate,
        num_blocks=getattr(config.model, 'num_blocks', 4),
        base_channels=getattr(config.model, 'base_channels', 32)
    )

    print(f"Model: {model.__class__.__name__}")
    model_info = model.get_model_info()
    print(f"Parameters: {model_info['num_parameters']:,}")
    print(f"Input size: {model_info['input_size']}")
    print()

    # Create trainer
    trainer = RPeakTrainer(model, device)

    # Configure training
    loss_config = {
        'name': config.training.loss_name,
        **config.training.loss_params
    }

    optimizer_config = {
        'name': config.training.optimizer_name,
        'lr': config.training.learning_rate,
        'weight_decay': config.training.weight_decay
    }

    scheduler_config = None
    if config.training.use_scheduler:
        scheduler_config = {
            'name': config.training.scheduler_name,
            **config.training.scheduler_params
        }

    trainer.configure_training(loss_config, optimizer_config, scheduler_config)
    print("Trainer configured successfully")
    print()

    return trainer


def run_training(trainer: RPeakTrainer, train_loader, val_loader, config: RPeakConfig):
    """Run the training process."""
    print("=== Training ===")

    with Timer("Training"):
        training_results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.training.epochs,
            save_dir=config.output.models_dir,
            early_stopping_patience=config.training.early_stopping_patience,
            early_stopping_metric=config.training.early_stopping_metric,
            save_best_only=config.training.save_best_only,
            verbose=config.verbose,
            log_every_epoch=True  # Enable detailed epoch-wise logging
        )

    print(f"Training completed!")
    print(f"Total epochs: {training_results['total_epochs']}")
    print(f"Best {config.training.early_stopping_metric}: {training_results['best_metric_value']:.4f}")
    print(f"Early stopped: {training_results['early_stopped']}")
    print()

    return training_results


def run_evaluation(trainer: RPeakTrainer, test_loader, config: RPeakConfig):
    """Run final evaluation on test set."""
    print("=== Final Evaluation ===")

    with Timer("Test evaluation"):
        test_metrics = trainer.evaluate(test_loader, threshold=config.evaluation.threshold)

    # Print test results
    from src.training.metrics import print_metrics
    print_metrics(test_metrics, "Test Results")

    return test_metrics


def main():
    """Main training pipeline."""
    args = parse_arguments()

    # Create default configs if requested
    if args.create_default_configs:
        create_default_configs()
        return

    # Load configuration
    config = load_config(args.config)
    apply_argument_overrides(config, args)

    # Validate configuration
    if not validate_config(config):
        print("Configuration validation failed. Please fix the issues and try again.")
        return

    # Print configuration
    if config.verbose:
        print_config(config)

    # Setup environment
    set_seed(config.seed)
    device = get_device_config(config.device)
    create_output_directories(config.output.base_dir)

    if config.verbose:
        print_system_info()

    # Save configuration used for this run
    config_save_path = os.path.join(config.output.base_dir, "configs", "run_config.yaml")
    save_config(config, config_save_path)

    try:
        # Load and prepare data
        data_results = load_and_prepare_data(config)
        eeg_train, eeg_val, eeg_test, rpeaks_train, rpeaks_val, rpeaks_test, scaler, stats = data_results

        if args.dry_run:
            print("Dry run completed successfully!")
            return

        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            config, eeg_train, eeg_val, eeg_test, rpeaks_train, rpeaks_val, rpeaks_test
        )

        # Setup training
        trainer = setup_training(config, device)

        # Run training
        training_results = run_training(trainer, train_loader, val_loader, config)

        # Create training curves plot
        if config.visualization.create_plots:
            with Timer("Training curves plot"):
                plot_training_curves(
                    training_results['metrics_history'],
                    save_path=os.path.join(config.output.plots_dir, "training_curves.png")
                )

        # Run final evaluation
        test_metrics = run_evaluation(trainer, test_loader, config)

        # Compile final results
        final_results = {
            'config': config.__dict__,
            'dataset_stats': stats,
            'training_results': training_results,
            'test_metrics': test_metrics,
            'model_info': trainer.model.get_model_info()
        }

        # Save results
        results_path = os.path.join(config.output.base_dir, "results.json")
        save_results(final_results, results_path)

        print("=== Training Pipeline Completed Successfully! ===")
        print(f"Results saved to: {config.output.base_dir}")
        print(f"Best model saved to: {config.output.models_dir}")
        if config.visualization.create_plots:
            print(f"Plots saved to: {config.output.plots_dir}")

    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
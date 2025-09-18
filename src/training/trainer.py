"""
Training utilities and trainer class for R-peak detection models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from typing import Dict, Optional, Callable, Any
import time
import json
import datetime
import csv

from .metrics import calculate_classification_metrics, MetricsTracker, print_metrics
from .losses import get_loss_function


class RPeakTrainer:
    """
    Trainer class for R-peak detection models.
    """

    def __init__(self, model: nn.Module, device: torch.device = None):
        """
        Initialize trainer.

        Args:
            model (nn.Module): Model to train
            device (torch.device): Device to train on
        """
        self.model = model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Training components (to be set via configure_training)
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        # Training state
        self.metrics_tracker = MetricsTracker()
        self.current_epoch = 0
        self.best_metric_value = 0.0
        self.patience_counter = 0

        print(f"Training on device: {self.device}")

    def configure_training(self, loss_config: Dict[str, Any], optimizer_config: Dict[str, Any],
                          scheduler_config: Optional[Dict[str, Any]] = None):
        """
        Configure training components.

        Args:
            loss_config (dict): Loss function configuration
            optimizer_config (dict): Optimizer configuration
            scheduler_config (dict, optional): Scheduler configuration
        """
        # Configure loss function
        loss_name = loss_config.pop('name')
        self.criterion = get_loss_function(loss_name, **loss_config)

        # Configure optimizer
        optimizer_name = optimizer_config.pop('name')
        lr = optimizer_config.pop('lr', 0.001)

        if optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, **optimizer_config)
        elif optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, **optimizer_config)
        elif optimizer_name.lower() == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, **optimizer_config)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Configure scheduler if provided
        if scheduler_config is not None:
            scheduler_name = scheduler_config.pop('name')
            if scheduler_name.lower() == 'reduce_on_plateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **scheduler_config)
            elif scheduler_name.lower() == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, **scheduler_config)
            elif scheduler_name.lower() == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **scheduler_config)
            else:
                raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader (DataLoader): Training data loader

        Returns:
            dict: Training metrics for the epoch
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_scores = []

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Store predictions for metrics calculation
            if hasattr(self.criterion, '__class__') and 'Logits' in self.criterion.__class__.__name__:
                # If using logits loss, apply sigmoid to get probabilities
                scores = torch.sigmoid(output)
            else:
                scores = output

            predictions = (scores > 0.5).float()

            all_predictions.extend(predictions.cpu().detach().numpy().flatten())
            all_targets.extend(target.cpu().detach().numpy().flatten())
            all_scores.extend(scores.cpu().detach().numpy().flatten())

        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        metrics = calculate_classification_metrics(
            np.array(all_targets), np.array(all_predictions), np.array(all_scores)
        )
        metrics['loss'] = avg_loss

        return metrics

    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate for one epoch.

        Args:
            val_loader (DataLoader): Validation data loader

        Returns:
            dict: Validation metrics for the epoch
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_scores = []

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()

                # Store predictions for metrics calculation
                if hasattr(self.criterion, '__class__') and 'Logits' in self.criterion.__class__.__name__:
                    scores = torch.sigmoid(output)
                else:
                    scores = output

                predictions = (scores > 0.5).float()

                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())
                all_scores.extend(scores.cpu().numpy().flatten())

        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        metrics = calculate_classification_metrics(
            np.array(all_targets), np.array(all_predictions), np.array(all_scores)
        )
        metrics['loss'] = avg_loss

        return metrics

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int, save_dir: str = 'outputs/models',
              early_stopping_patience: int = 10, early_stopping_metric: str = 'f1_score',
              save_best_only: bool = True, verbose: bool = True,
              log_every_epoch: bool = True) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            epochs (int): Number of epochs to train
            save_dir (str): Directory to save models
            early_stopping_patience (int): Patience for early stopping
            early_stopping_metric (str): Metric to monitor for early stopping
            save_best_only (bool): Whether to save only the best model
            verbose (bool): Whether to print training progress
            log_every_epoch (bool): Whether to log detailed metrics every epoch

        Returns:
            dict: Training history and results
        """
        os.makedirs(save_dir, exist_ok=True)
        best_model_path = os.path.join(save_dir, 'best_model.pth')

        # Setup logging if requested
        if log_every_epoch:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join(save_dir, '../logs')
            os.makedirs(log_dir, exist_ok=True)

            epoch_log_file = os.path.join(log_dir, f'trainer_log_{timestamp}.csv')
            config_log_file = os.path.join(log_dir, f'trainer_config_{timestamp}.json')

            # Initialize CSV log for epoch-wise data
            with open(epoch_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'train_precision', 'train_recall',
                               'train_f1_score', 'train_specificity', 'train_auc_roc', 'train_auc_pr',
                               'val_loss', 'val_accuracy', 'val_precision', 'val_recall',
                               'val_f1_score', 'val_specificity', 'val_auc_roc', 'val_auc_pr',
                               'learning_rate', 'epoch_time', 'patience_counter'])

            # Log training configuration
            config_data = {
                'timestamp': timestamp,
                'training_params': {
                    'epochs': epochs,
                    'early_stopping_patience': early_stopping_patience,
                    'early_stopping_metric': early_stopping_metric,
                    'save_best_only': save_best_only
                },
                'optimizer_config': {
                    'type': self.optimizer.__class__.__name__ if self.optimizer else None,
                    'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer else None,
                    'params': {k: v for k, v in self.optimizer.param_groups[0].items()
                              if k != 'params'} if self.optimizer else None
                },
                'scheduler_config': {
                    'type': self.scheduler.__class__.__name__ if self.scheduler else None
                },
                'criterion_config': {
                    'type': self.criterion.__class__.__name__ if self.criterion else None
                },
                'model_info': {
                    'total_params': sum(p.numel() for p in self.model.parameters()),
                    'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                },
                'device': str(self.device),
                'data_info': {
                    'train_batches': len(train_loader),
                    'val_batches': len(val_loader),
                    'train_batch_size': train_loader.batch_size,
                    'val_batch_size': val_loader.batch_size
                }
            }

            with open(config_log_file, 'w') as f:
                json.dump(config_data, f, indent=2)

            if verbose:
                print(f"Training configuration saved to: {config_log_file}")
                print(f"Epoch-wise logs will be saved to: {epoch_log_file}")

        self.metrics_tracker.reset()
        self.best_metric_value = 0.0
        self.patience_counter = 0

        start_time = time.time()

        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Training phase
            train_metrics = self.train_epoch(train_loader)
            self.metrics_tracker.update(train_metrics, 'train')

            # Validation phase
            val_metrics = self.validate_epoch(val_loader)
            self.metrics_tracker.update(val_metrics, 'val')

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics[early_stopping_metric])
                else:
                    self.scheduler.step()

            # Early stopping and model saving
            current_metric = val_metrics[early_stopping_metric]
            if current_metric > self.best_metric_value:
                self.best_metric_value = current_metric
                self.patience_counter = 0

                if save_best_only:
                    self.save_model(best_model_path)
            else:
                self.patience_counter += 1

            # Log epoch data to CSV if logging is enabled
            if log_every_epoch:
                epoch_time = time.time() - epoch_start_time
                current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0

                with open(epoch_log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch,
                        train_metrics.get('loss', 0.0),
                        train_metrics.get('accuracy', 0.0),
                        train_metrics.get('precision', 0.0),
                        train_metrics.get('recall', 0.0),
                        train_metrics.get('f1_score', 0.0),
                        train_metrics.get('specificity', 0.0),
                        train_metrics.get('auc_roc', 0.0),
                        train_metrics.get('auc_pr', 0.0),
                        val_metrics.get('loss', 0.0),
                        val_metrics.get('accuracy', 0.0),
                        val_metrics.get('precision', 0.0),
                        val_metrics.get('recall', 0.0),
                        val_metrics.get('f1_score', 0.0),
                        val_metrics.get('specificity', 0.0),
                        val_metrics.get('auc_roc', 0.0),
                        val_metrics.get('auc_pr', 0.0),
                        current_lr,
                        epoch_time,
                        self.patience_counter
                    ])

            # Print progress
            if verbose and (epoch % 5 == 0 or epoch < 10):
                epoch_time = time.time() - epoch_start_time
                print(f'Epoch {epoch:3d}: '
                      f'Train Loss: {train_metrics["loss"]:.4f}, '
                      f'Val Loss: {val_metrics["loss"]:.4f}, '
                      f'Train F1: {train_metrics["f1_score"]:.3f}, '
                      f'Val F1: {val_metrics["f1_score"]:.3f}, '
                      f'Time: {epoch_time:.1f}s')

            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        total_time = time.time() - start_time

        # Load best model if it exists
        if os.path.exists(best_model_path):
            self.load_model(best_model_path)
            if verbose:
                print("Loaded best model from checkpoint")

        # Final results
        results = {
            'metrics_history': self.metrics_tracker.get_history(),
            'best_metric_value': self.best_metric_value,
            'total_epochs': epoch + 1,
            'total_time': total_time,
            'early_stopped': self.patience_counter >= early_stopping_patience
        }

        if verbose:
            print(f"\nTraining completed in {total_time:.1f}s")
            print(f"Best {early_stopping_metric}: {self.best_metric_value:.4f}")
            if log_every_epoch:
                print(f"Detailed logs saved to: {epoch_log_file}")

        return results

    def evaluate(self, test_loader: DataLoader, threshold: float = 0.5) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            test_loader (DataLoader): Test data loader
            threshold (float): Classification threshold

        Returns:
            dict: Test metrics
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_scores = []

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                output = self.model(data)

                if hasattr(self.criterion, '__class__') and 'Logits' in self.criterion.__class__.__name__:
                    scores = torch.sigmoid(output)
                else:
                    scores = output

                predictions = (scores > threshold).float()

                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())
                all_scores.extend(scores.cpu().numpy().flatten())

        # Calculate comprehensive metrics
        metrics = calculate_classification_metrics(
            np.array(all_targets), np.array(all_predictions), np.array(all_scores)
        )

        return metrics

    def save_model(self, filepath: str):
        """
        Save model checkpoint.

        Args:
            filepath (str): Path to save the model
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': self.current_epoch,
            'best_metric_value': self.best_metric_value,
            'metrics_history': self.metrics_tracker.get_history(),
            'timestamp': datetime.datetime.now().isoformat(),
            'model_info': {
                'total_params': sum(p.numel() for p in self.model.parameters()),
                'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            }
        }
        torch.save(checkpoint, filepath)

    def load_model(self, filepath: str, load_optimizer: bool = False):
        """
        Load model checkpoint.

        Args:
            filepath (str): Path to the saved model
            load_optimizer (bool): Whether to load optimizer state
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if load_optimizer and self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_metric_value = checkpoint.get('best_metric_value', 0.0)

        if 'metrics_history' in checkpoint:
            self.metrics_tracker.metrics_history = checkpoint['metrics_history']
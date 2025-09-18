"""
Evaluation metrics for R-peak detection.
"""

import numpy as np
import torch
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    accuracy_score
)
from typing import Dict, Tuple, Optional, Union


def calculate_confusion_matrix_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate metrics from confusion matrix.

    Args:
        y_true (np.ndarray): Ground truth labels
        y_pred (np.ndarray): Predicted labels (binary)

    Returns:
        dict: Dictionary containing TP, TN, FP, FN, sensitivity, specificity
    """
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle case where only one class is present
        if len(np.unique(y_true)) == 1:
            if y_true[0] == 0:  # Only negative class
                tn = len(y_true) - np.sum(y_pred)
                fp = np.sum(y_pred)
                fn = 0
                tp = 0
            else:  # Only positive class
                tn = 0
                fp = 0
                fn = len(y_true) - np.sum(y_pred)
                tp = np.sum(y_pred)
        else:
            raise ValueError("Unexpected confusion matrix shape")

    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value

    return {
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'sensitivity': sensitivity, 'specificity': specificity,
        'precision': precision, 'npv': npv
    }


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                   y_scores: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true (np.ndarray): Ground truth labels
        y_pred (np.ndarray): Predicted labels (binary)
        y_scores (np.ndarray, optional): Prediction scores/probabilities

    Returns:
        dict: Dictionary containing all calculated metrics
    """
    # Basic metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)

    # Confusion matrix metrics
    cm_metrics = calculate_confusion_matrix_metrics(y_true, y_pred)

    # Combine metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'sensitivity': cm_metrics['sensitivity'],  # Same as recall
        'specificity': cm_metrics['specificity'],
        **cm_metrics
    }

    # AUC metrics if scores are provided
    if y_scores is not None:
        try:
            if len(np.unique(y_true)) > 1:  # Need both classes for AUC
                metrics['auc_roc'] = roc_auc_score(y_true, y_scores)
                metrics['auc_pr'] = average_precision_score(y_true, y_scores)
            else:
                metrics['auc_roc'] = 0.0
                metrics['auc_pr'] = 0.0
        except ValueError:
            metrics['auc_roc'] = 0.0
            metrics['auc_pr'] = 0.0

    return metrics


def evaluate_at_multiple_thresholds(y_true: np.ndarray, y_scores: np.ndarray,
                                  thresholds: list = None) -> Dict[float, Dict[str, float]]:
    """
    Evaluate metrics at multiple prediction thresholds.

    Args:
        y_true (np.ndarray): Ground truth labels
        y_scores (np.ndarray): Prediction scores
        thresholds (list): List of thresholds to evaluate

    Returns:
        dict: Dictionary with threshold as key and metrics as values
    """
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    results = {}

    for threshold in thresholds:
        y_pred = (y_scores > threshold).astype(int)
        metrics = calculate_classification_metrics(y_true, y_pred, y_scores)
        results[threshold] = metrics

    return results


def find_optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray,
                          metric: str = 'f1_score', n_thresholds: int = 100) -> Tuple[float, float]:
    """
    Find optimal threshold based on specified metric.

    Args:
        y_true (np.ndarray): Ground truth labels
        y_scores (np.ndarray): Prediction scores
        metric (str): Metric to optimize ('f1_score', 'precision', 'recall', etc.)
        n_thresholds (int): Number of thresholds to test

    Returns:
        tuple: (optimal_threshold, best_metric_value)
    """
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    best_threshold = 0.5
    best_metric_value = 0.0

    for threshold in thresholds:
        y_pred = (y_scores > threshold).astype(int)
        metrics = calculate_classification_metrics(y_true, y_pred, y_scores)

        if metrics[metric] > best_metric_value:
            best_metric_value = metrics[metric]
            best_threshold = threshold

    return best_threshold, best_metric_value


class MetricsTracker:
    """
    Class to track metrics during training.
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics_history = {}

    def update(self, metrics: Dict[str, float], phase: str = 'train'):
        """
        Update metrics for a given phase.

        Args:
            metrics (dict): Dictionary of metric values
            phase (str): Training phase ('train', 'val', 'test')
        """
        if phase not in self.metrics_history:
            self.metrics_history[phase] = {}

        for metric_name, value in metrics.items():
            full_name = f"{phase}_{metric_name}"
            if full_name not in self.metrics_history[phase]:
                self.metrics_history[phase][full_name] = []
            self.metrics_history[phase][full_name].append(value)

    def get_latest(self, metric_name: str, phase: str = 'train') -> float:
        """
        Get latest value of a metric.

        Args:
            metric_name (str): Name of the metric
            phase (str): Training phase

        Returns:
            float: Latest metric value
        """
        full_name = f"{phase}_{metric_name}"
        if phase in self.metrics_history and full_name in self.metrics_history[phase]:
            return self.metrics_history[phase][full_name][-1]
        return 0.0

    def get_best(self, metric_name: str, phase: str = 'val', mode: str = 'max') -> Tuple[float, int]:
        """
        Get best value of a metric and its epoch.

        Args:
            metric_name (str): Name of the metric
            phase (str): Training phase
            mode (str): 'max' for higher is better, 'min' for lower is better

        Returns:
            tuple: (best_value, epoch)
        """
        full_name = f"{phase}_{metric_name}"
        if phase not in self.metrics_history or full_name not in self.metrics_history[phase]:
            return 0.0, 0

        values = self.metrics_history[phase][full_name]
        if mode == 'max':
            best_idx = np.argmax(values)
            best_value = values[best_idx]
        else:
            best_idx = np.argmin(values)
            best_value = values[best_idx]

        return best_value, best_idx

    def get_history(self) -> Dict:
        """Get complete metrics history."""
        return self.metrics_history

    def reset(self):
        """Reset metrics history."""
        self.metrics_history = {}


def print_metrics(metrics: Dict[str, float], title: str = "Metrics", precision: int = 4):
    """
    Print metrics in a formatted way.

    Args:
        metrics (dict): Dictionary of metrics
        title (str): Title for the metrics display
        precision (int): Number of decimal places
    """
    print(f"\n=== {title} ===")

    # Group metrics by type
    basic_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'sensitivity', 'specificity']
    confusion_metrics = ['tp', 'tn', 'fp', 'fn']
    auc_metrics = ['auc_roc', 'auc_pr']

    # Print basic metrics
    for metric in basic_metrics:
        if metric in metrics:
            print(f"{metric.replace('_', ' ').title():<12}: {metrics[metric]:.{precision}f}")

    # Print confusion matrix metrics
    if any(metric in metrics for metric in confusion_metrics):
        print("\nConfusion Matrix:")
        if all(metric in metrics for metric in confusion_metrics):
            print(f"TP: {metrics['tp']:>6d}  FN: {metrics['fn']:>6d}")
            print(f"FP: {metrics['fp']:>6d}  TN: {metrics['tn']:>6d}")

    # Print AUC metrics
    for metric in auc_metrics:
        if metric in metrics:
            print(f"{metric.replace('_', ' ').upper():<12}: {metrics[metric]:.{precision}f}")


def format_metrics_for_logging(metrics: Dict[str, float], prefix: str = "") -> str:
    """
    Format metrics for logging.

    Args:
        metrics (dict): Dictionary of metrics
        prefix (str): Prefix for metric names

    Returns:
        str: Formatted metrics string
    """
    formatted_parts = []
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            if isinstance(value, int):
                formatted_parts.append(f"{prefix}{name}: {value}")
            else:
                formatted_parts.append(f"{prefix}{name}: {value:.4f}")

    return ", ".join(formatted_parts)
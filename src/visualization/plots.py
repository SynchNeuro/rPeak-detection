"""
Plotting and visualization utilities for ECG/EEG analysis and R-peak detection.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Any, Tuple
import os


def create_ecg_validation_plot(ecg: np.ndarray, rpeak_binary: np.ndarray,
                             rpeak_locations: np.ndarray, eeg_filtered: Optional[np.ndarray] = None,
                             sampling_rate: float = 250.0, duration: float = 30.0,
                             save_path: str = 'outputs/plots/ecg_validation.png') -> Optional[float]:
    """
    Create detailed ECG validation plot with R-peaks and EEG overlay.

    Args:
        ecg (np.ndarray): ECG signal
        rpeak_binary (np.ndarray): Binary R-peak signal
        rpeak_locations (np.ndarray): R-peak sample indices
        eeg_filtered (np.ndarray, optional): Filtered EEG signal
        sampling_rate (float): Signal sampling rate
        duration (float): Duration to plot in seconds
        save_path (str): Path to save the plot

    Returns:
        float: Calculated heart rate, or None if cannot be calculated
    """
    print("Creating detailed ECG validation plot...")

    samples_to_plot = min(int(duration * sampling_rate), len(ecg))
    time_axis = np.arange(samples_to_plot) / sampling_rate

    fig, axes = plt.subplots(3 if eeg_filtered is not None else 2, 1, figsize=(15, 12))

    # Plot 1: ECG with R-peaks
    axes[0].plot(time_axis, ecg[:samples_to_plot], 'b-', linewidth=1, label='ECG Signal')
    rpeak_indices_in_window = rpeak_locations[rpeak_locations < samples_to_plot]
    axes[0].scatter(rpeak_indices_in_window / sampling_rate, ecg[rpeak_indices_in_window],
                    color='red', s=50, zorder=5, label=f'R-peaks ({len(rpeak_indices_in_window)})')
    axes[0].set_title('ECG Signal with Detected R-peaks (Ground Truth Verification)', fontsize=14)
    axes[0].set_ylabel('ECG Amplitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Binary R-peak signal
    axes[1].plot(time_axis, rpeak_binary[:samples_to_plot], 'r-', linewidth=2, label='R-peak Events')
    axes[1].set_title('Binary R-peak Signal (Ground Truth Labels)', fontsize=14)
    axes[1].set_ylabel('R-peak (1/0)')
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Filtered EEG if provided
    if eeg_filtered is not None:
        axes[2].plot(time_axis, eeg_filtered[:samples_to_plot], 'g-', linewidth=1, label='Filtered EEG (1-35Hz)')
        for i, rpeak_idx in enumerate(rpeak_indices_in_window[:5]):
            axes[2].axvline(x=rpeak_idx / sampling_rate, color='red', linestyle='--', alpha=0.5,
                           label='R-peak timing' if i == 0 else '')
        axes[2].set_title('Filtered EEG Signal (Input) with R-peak Timing', fontsize=14)
        axes[2].set_ylabel('EEG Amplitude')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')

    # Calculate and display heart rate statistics
    heart_rate = None
    if len(rpeak_indices_in_window) > 1:
        rr_intervals = np.diff(rpeak_indices_in_window) / sampling_rate
        heart_rate = 60 / np.mean(rr_intervals)
        fig.suptitle(f'ECG Analysis - Heart Rate: {heart_rate:.1f} BPM, '
                     f'RR Interval: {np.mean(rr_intervals)*1000:.1f}±{np.std(rr_intervals)*1000:.1f}ms',
                     fontsize=16, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"ECG validation plot saved as '{save_path}'")
    return heart_rate


def plot_training_curves(metrics_history: Dict[str, List[float]],
                        save_path: str = 'outputs/plots/training_curves.png'):
    """
    Plot comprehensive training curves with all metrics.

    Args:
        metrics_history (dict): Dictionary containing training metrics history
        save_path (str): Path to save the plot
    """
    print("Creating comprehensive training curves...")

    # Extract data for each phase
    train_data = {}
    val_data = {}

    for phase, metrics in metrics_history.items():
        for metric_name, values in metrics.items():
            clean_name = metric_name.replace(f'{phase}_', '')
            if phase == 'train':
                train_data[clean_name] = values
            elif phase == 'val':
                val_data[clean_name] = values

    if not train_data and not val_data:
        print("No metrics data found for plotting")
        return

    # Determine number of epochs
    epochs = range(1, len(list(train_data.values())[0]) + 1) if train_data else range(1, len(list(val_data.values())[0]) + 1)

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    # Define metrics to plot
    metrics_to_plot = [
        ('loss', 'Loss'),
        ('precision', 'Precision (PPV)'),
        ('recall', 'Recall (Sensitivity)'),
        ('f1_score', 'F1 Score'),
        ('specificity', 'Specificity'),
        ('accuracy', 'Accuracy')
    ]

    for idx, (metric_key, metric_title) in enumerate(metrics_to_plot):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Plot training data
        if metric_key in train_data:
            ax.plot(epochs, train_data[metric_key], 'b-', label=f'Training {metric_title}', linewidth=2)

        # Plot validation data
        if metric_key in val_data:
            ax.plot(epochs, val_data[metric_key], 'r-', label=f'Validation {metric_title}', linewidth=2)

        ax.set_title(metric_title, fontsize=14)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Comprehensive Training Metrics for R-peak Detection', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Training curves saved as '{save_path}'")


def plot_prediction_results(eeg_signal: np.ndarray, true_rpeaks: np.ndarray,
                           predicted_rpeaks: np.ndarray, prediction_scores: Optional[np.ndarray] = None,
                           sampling_rate: float = 250.0, duration: float = 20.0,
                           save_path: str = 'outputs/plots/prediction_results.png'):
    """
    Create comprehensive results visualization showing EEG, true R-peaks, and predictions.

    Args:
        eeg_signal (np.ndarray): EEG signal
        true_rpeaks (np.ndarray): Ground truth R-peak labels
        predicted_rpeaks (np.ndarray): Predicted R-peak labels
        prediction_scores (np.ndarray, optional): Prediction confidence scores
        sampling_rate (float): Signal sampling rate
        duration (float): Duration to plot in seconds
        save_path (str): Path to save the plot
    """
    print("Creating comprehensive results visualization...")

    window_size = 125  # Standard window size
    center_offset = window_size // 2
    samples_to_plot = min(int(duration * sampling_rate), len(predicted_rpeaks))
    time_axis = np.arange(samples_to_plot) / sampling_rate

    # Adjust signals for windowing
    eeg_plot = eeg_signal[center_offset:center_offset + samples_to_plot]
    rpeaks_plot = true_rpeaks[center_offset:center_offset + samples_to_plot]
    pred_plot = predicted_rpeaks[:samples_to_plot]

    n_plots = 4 if prediction_scores is not None else 3
    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 4*n_plots))

    plot_idx = 0

    # Plot 1: EEG signal
    axes[plot_idx].plot(time_axis, eeg_plot, 'b-', linewidth=1, alpha=0.8)
    axes[plot_idx].set_title('Filtered EEG Signal (1-35Hz)', fontsize=14)
    axes[plot_idx].set_ylabel('EEG Amplitude')
    axes[plot_idx].grid(True, alpha=0.3)
    plot_idx += 1

    # Plot 2: True R-peaks
    axes[plot_idx].plot(time_axis, rpeaks_plot, 'g-', linewidth=2, label='Ground Truth')
    true_rpeak_indices = np.where(np.array(rpeaks_plot) == 1)[0]
    if len(true_rpeak_indices) > 0:
        axes[plot_idx].scatter(true_rpeak_indices / sampling_rate, np.ones(len(true_rpeak_indices)),
                              color='green', s=50, zorder=5, label=f'True R-peaks ({len(true_rpeak_indices)})')
    axes[plot_idx].set_title('Ground Truth R-peaks (from ECG)', fontsize=14)
    axes[plot_idx].set_ylabel('R-peak')
    axes[plot_idx].set_ylim(-0.1, 1.2)
    axes[plot_idx].legend()
    axes[plot_idx].grid(True, alpha=0.3)
    plot_idx += 1

    # Plot 3: Prediction scores (if available)
    if prediction_scores is not None:
        score_plot = prediction_scores[:samples_to_plot]
        axes[plot_idx].plot(time_axis, score_plot, 'orange', linewidth=1, alpha=0.8, label='Prediction Score')
        axes[plot_idx].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Threshold (0.5)')
        axes[plot_idx].set_title('Model Prediction Scores', fontsize=14)
        axes[plot_idx].set_ylabel('Score')
        axes[plot_idx].set_ylim(0, 1)
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1

    # Plot 4: Predicted R-peaks vs True R-peaks comparison
    axes[plot_idx].plot(time_axis, rpeaks_plot, 'g-', linewidth=2, alpha=0.7, label='True R-peaks')
    axes[plot_idx].plot(time_axis, pred_plot, 'r-', linewidth=2, alpha=0.7, label='Predicted R-peaks')

    # Mark correct and incorrect predictions
    correct_predictions = np.array(rpeaks_plot) == np.array(pred_plot)
    incorrect_indices = np.where(~correct_predictions)[0]
    if len(incorrect_indices) > 0:
        axes[plot_idx].scatter(incorrect_indices / sampling_rate, np.array(pred_plot)[incorrect_indices],
                              color='red', s=30, marker='x', zorder=5, alpha=0.8, label='Errors')

    axes[plot_idx].set_title('Prediction Comparison', fontsize=14)
    axes[plot_idx].set_xlabel('Time (s)')
    axes[plot_idx].set_ylabel('R-peak')
    axes[plot_idx].set_ylim(-0.1, 1.2)
    axes[plot_idx].legend()
    axes[plot_idx].grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Prediction results saved as '{save_path}'")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         save_path: str = 'outputs/plots/confusion_matrix.png'):
    """
    Plot confusion matrix.

    Args:
        y_true (np.ndarray): Ground truth labels
        y_pred (np.ndarray): Predicted labels
        save_path (str): Path to save the plot
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No R-peak', 'R-peak'],
                yticklabels=['No R-peak', 'R-peak'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrix saved as '{save_path}'")


def plot_metrics_comparison(results_dict: Dict[str, Dict[str, float]],
                           save_path: str = 'outputs/plots/metrics_comparison.png'):
    """
    Plot comparison of metrics across different models or configurations.

    Args:
        results_dict (dict): Dictionary with model names as keys and metrics as values
        save_path (str): Path to save the plot
    """
    if not results_dict:
        print("No results provided for comparison")
        return

    # Extract metrics and model names
    model_names = list(results_dict.keys())
    metrics_names = ['precision', 'recall', 'f1_score', 'specificity', 'accuracy']

    # Create comparison data
    metrics_data = {metric: [] for metric in metrics_names}
    for model_name in model_names:
        for metric in metrics_names:
            metrics_data[metric].append(results_dict[model_name].get(metric, 0))

    # Create bar plot
    x = np.arange(len(model_names))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, metric in enumerate(metrics_names):
        ax.bar(x + i * width, metrics_data[metric], width, label=metric.replace('_', ' ').title())

    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Metrics comparison saved as '{save_path}'")
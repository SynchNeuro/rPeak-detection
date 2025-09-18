"""
General helper utilities for the R-peak detection project.
"""

import torch
import numpy as np
import random
import os
import json
import pickle
from typing import Any, Dict, List, Optional
import time
from datetime import datetime


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_output_directories(base_dir: str = "outputs"):
    """
    Create necessary output directories.

    Args:
        base_dir (str): Base output directory
    """
    dirs_to_create = [
        base_dir,
        os.path.join(base_dir, "models"),
        os.path.join(base_dir, "plots"),
        os.path.join(base_dir, "logs"),
        os.path.join(base_dir, "predictions"),
        os.path.join(base_dir, "configs")
    ]

    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)

    print(f"Created output directories in '{base_dir}'")


def save_results(results: Dict[str, Any], filepath: str):
    """
    Save results to file in JSON format.

    Args:
        results (dict): Results dictionary
        filepath (str): Path to save file
    """
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj

    results_serializable = convert_numpy(results)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"Results saved to '{filepath}'")


def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load results from JSON file.

    Args:
        filepath (str): Path to results file

    Returns:
        dict: Loaded results
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Results file not found: {filepath}")

    with open(filepath, 'r') as f:
        results = json.load(f)

    return results


def save_predictions(predictions: np.ndarray, scores: np.ndarray,
                    true_labels: np.ndarray, filepath: str):
    """
    Save model predictions.

    Args:
        predictions (np.ndarray): Binary predictions
        scores (np.ndarray): Prediction scores
        true_labels (np.ndarray): Ground truth labels
        filepath (str): Path to save file
    """
    data = {
        'predictions': predictions,
        'scores': scores,
        'true_labels': true_labels,
        'timestamp': datetime.now().isoformat()
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

    print(f"Predictions saved to '{filepath}'")


def load_predictions(filepath: str) -> Dict[str, np.ndarray]:
    """
    Load saved predictions.

    Args:
        filepath (str): Path to predictions file

    Returns:
        dict: Loaded predictions data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Predictions file not found: {filepath}")

    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    return data


def get_model_size(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Get model size information.

    Args:
        model (torch.nn.Module): PyTorch model

    Returns:
        dict: Model size information
    """
    param_size = 0
    param_count = 0
    buffer_size = 0
    buffer_count = 0

    for param in model.parameters():
        param_count += param.numel()
        param_size += param.numel() * param.element_size()

    for buffer in model.buffers():
        buffer_count += buffer.numel()
        buffer_size += buffer.numel() * buffer.element_size()

    total_size = param_size + buffer_size
    size_mb = total_size / 1024 / 1024

    return {
        'param_count': param_count,
        'buffer_count': buffer_count,
        'total_params': param_count + buffer_count,
        'param_size_bytes': param_size,
        'buffer_size_bytes': buffer_size,
        'total_size_bytes': total_size,
        'total_size_mb': size_mb
    }


def format_time(seconds: float) -> str:
    """
    Format time duration in a human-readable way.

    Args:
        seconds (float): Time in seconds

    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours}h {minutes}m {remaining_seconds:.1f}s"


def print_system_info():
    """Print system information for debugging."""
    print("=== System Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Number of CPU cores: {torch.get_num_threads()}")
    print()


class Timer:
    """Simple timer context manager."""

    def __init__(self, description: str = "Operation"):
        """
        Initialize timer.

        Args:
            description (str): Description of the operation being timed
        """
        self.description = description
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        print(f"Starting {self.description}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and print duration."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"{self.description} completed in {format_time(duration)}")

    def elapsed(self) -> float:
        """
        Get elapsed time.

        Returns:
            float: Elapsed time in seconds
        """
        if self.start_time is None:
            return 0.0
        current_time = time.time() if self.end_time is None else self.end_time
        return current_time - self.start_time


def check_data_quality(eeg_data: List[np.ndarray], rpeak_data: List[np.ndarray]) -> Dict[str, Any]:
    """
    Check data quality and provide statistics.

    Args:
        eeg_data (List[np.ndarray]): EEG data arrays
        rpeak_data (List[np.ndarray]): R-peak data arrays

    Returns:
        dict: Data quality report
    """
    quality_report = {
        'num_files': len(eeg_data),
        'total_samples': sum(len(eeg) for eeg in eeg_data),
        'total_rpeaks': sum(np.sum(rpeak) for rpeak in rpeak_data),
        'file_stats': [],
        'issues': []
    }

    for i, (eeg, rpeak) in enumerate(zip(eeg_data, rpeak_data)):
        file_stats = {
            'file_index': i,
            'samples': len(eeg),
            'rpeaks': int(np.sum(rpeak)),
            'rpeak_percentage': np.sum(rpeak) / len(rpeak) * 100,
            'eeg_mean': float(np.mean(eeg)),
            'eeg_std': float(np.std(eeg)),
            'has_nan': bool(np.isnan(eeg).any() or np.isnan(rpeak).any()),
            'has_inf': bool(np.isinf(eeg).any() or np.isinf(rpeak).any())
        }

        quality_report['file_stats'].append(file_stats)

        # Check for issues
        if file_stats['has_nan']:
            quality_report['issues'].append(f"File {i}: Contains NaN values")

        if file_stats['has_inf']:
            quality_report['issues'].append(f"File {i}: Contains infinite values")

        if file_stats['rpeaks'] == 0:
            quality_report['issues'].append(f"File {i}: No R-peaks detected")

        if file_stats['rpeak_percentage'] > 5.0:
            quality_report['issues'].append(f"File {i}: Unusually high R-peak percentage ({file_stats['rpeak_percentage']:.2f}%)")

    # Overall statistics
    quality_report['overall_rpeak_percentage'] = quality_report['total_rpeaks'] / quality_report['total_samples'] * 100
    quality_report['avg_samples_per_file'] = quality_report['total_samples'] / quality_report['num_files']
    quality_report['avg_rpeaks_per_file'] = quality_report['total_rpeaks'] / quality_report['num_files']

    return quality_report


def print_data_quality_report(quality_report: Dict[str, Any]):
    """
    Print data quality report in a formatted way.

    Args:
        quality_report (dict): Data quality report from check_data_quality
    """
    print("=== Data Quality Report ===")
    print(f"Number of files: {quality_report['num_files']}")
    print(f"Total samples: {quality_report['total_samples']:,}")
    print(f"Total R-peaks: {quality_report['total_rpeaks']:,}")
    print(f"Overall R-peak percentage: {quality_report['overall_rpeak_percentage']:.3f}%")
    print(f"Average samples per file: {quality_report['avg_samples_per_file']:.0f}")
    print(f"Average R-peaks per file: {quality_report['avg_rpeaks_per_file']:.0f}")

    if quality_report['issues']:
        print(f"\nIssues found:")
        for issue in quality_report['issues']:
            print(f"  - {issue}")
    else:
        print(f"\nNo data quality issues found!")

    print()


def ensure_reproducibility(seed: int = 42):
    """
    Ensure reproducibility by setting all relevant random seeds and configurations.

    Args:
        seed (int): Random seed
    """
    print(f"Setting random seed: {seed}")
    set_seed(seed)

    # Additional PyTorch settings for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)

    if torch.cuda.is_available():
        # These settings may impact performance but ensure reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    print("Reproducibility settings configured")
"""
PyTorch dataset classes for ECG/EEG R-peak detection.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, WeightedRandomSampler
from typing import Tuple, Optional


class ECGEEGDataset(Dataset):
    """
    Dataset class for EEG-based R-peak detection using windowed data.

    This dataset creates sliding windows of EEG data and uses the R-peak
    label at the center of each window as the target.
    """

    def __init__(self, eeg_data: np.ndarray, labels: np.ndarray, window_size: int = 125):
        """
        Initialize the dataset.

        Args:
            eeg_data (np.ndarray): EEG signal data
            labels (np.ndarray): Binary R-peak labels
            window_size (int): Size of the sliding window (samples)
        """
        self.eeg_data = eeg_data
        self.labels = labels
        self.window_size = window_size

        # Ensure we have enough data for at least one window
        if len(labels) < window_size:
            raise ValueError(f"Data length ({len(labels)}) is smaller than window size ({window_size})")

    def __len__(self) -> int:
        """Return the number of available windows."""
        return len(self.labels) - self.window_size + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a data sample.

        Args:
            idx (int): Sample index

        Returns:
            tuple: (eeg_window, label) where eeg_window is the EEG data window
                   and label is the R-peak label at the center of the window
        """
        # Extract EEG window
        eeg_window = self.eeg_data[idx:idx + self.window_size]

        # Label corresponds to the center of the window
        center_idx = idx + self.window_size // 2
        label = self.labels[center_idx]

        return torch.FloatTensor(eeg_window), torch.FloatTensor([label])


class BalancedECGEEGDataset(ECGEEGDataset):
    """
    Extended dataset class with support for class balancing.
    Inherits from ECGEEGDataset and adds balancing utilities.
    """

    def __init__(self, eeg_data: np.ndarray, labels: np.ndarray, window_size: int = 125):
        """
        Initialize the balanced dataset.

        Args:
            eeg_data (np.ndarray): EEG signal data
            labels (np.ndarray): Binary R-peak labels
            window_size (int): Size of the sliding window (samples)
        """
        super().__init__(eeg_data, labels, window_size)
        self._calculate_class_weights()

    def _calculate_class_weights(self):
        """Calculate class weights for balancing."""
        # Adjust labels for windowing (only consider labels that will be used as targets)
        effective_labels = self.labels[self.window_size//2:-(self.window_size//2)]

        # Calculate class distribution
        class_counts = np.bincount(effective_labels.astype(int))
        self.class_weights = 1.0 / class_counts

        self.pos_count = class_counts[1] if len(class_counts) > 1 else 0
        self.neg_count = class_counts[0]
        self.total_count = len(effective_labels)

    def get_class_distribution(self) -> dict:
        """
        Get information about class distribution.

        Returns:
            dict: Class distribution statistics
        """
        return {
            'positive_samples': int(self.pos_count),
            'negative_samples': int(self.neg_count),
            'total_samples': int(self.total_count),
            'positive_ratio': self.pos_count / self.total_count,
            'imbalance_ratio': self.neg_count / self.pos_count if self.pos_count > 0 else float('inf'),
            'class_weights': {
                'negative': self.class_weights[0],
                'positive': self.class_weights[1] if len(self.class_weights) > 1 else 0.0
            }
        }

    def create_balanced_sampler(self, replacement: bool = True) -> WeightedRandomSampler:
        """
        Create a balanced sampler for training.

        Args:
            replacement (bool): Whether to sample with replacement

        Returns:
            WeightedRandomSampler: Sampler for balanced training
        """
        # Calculate sample weights for each data point
        effective_labels = self.labels[self.window_size//2:-self.window_size//2+1]
        sample_weights = self.class_weights[effective_labels.astype(int)]

        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=replacement
        )


def create_datasets(eeg_train: np.ndarray, eeg_val: np.ndarray, eeg_test: np.ndarray,
                   rpeaks_train: np.ndarray, rpeaks_val: np.ndarray, rpeaks_test: np.ndarray,
                   window_size: int = 125, balanced: bool = True) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create train, validation, and test datasets.

    Args:
        eeg_train, eeg_val, eeg_test: EEG data splits
        rpeaks_train, rpeaks_val, rpeaks_test: R-peak label splits
        window_size (int): Window size for the dataset
        balanced (bool): Whether to use balanced dataset class

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    dataset_class = BalancedECGEEGDataset if balanced else ECGEEGDataset

    train_dataset = dataset_class(eeg_train, rpeaks_train, window_size)
    val_dataset = dataset_class(eeg_val, rpeaks_val, window_size)
    test_dataset = dataset_class(eeg_test, rpeaks_test, window_size)

    return train_dataset, val_dataset, test_dataset
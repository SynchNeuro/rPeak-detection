"""
Data loading utilities for OpenBCI ECG/EEG data files.
"""

import numpy as np
import pandas as pd
import glob
import os
from typing import Tuple, List, Optional

from ..preprocessing.signal_processing import (
    apply_bandpass_filter,
    detect_r_peaks
)


def load_openbci_file(file_path: str, max_samples: Optional[int] = 50000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ECG and EEG data from OpenBCI format file.

    Args:
        file_path (str): Path to the OpenBCI data file
        max_samples (int, optional): Maximum number of samples to load

    Returns:
        tuple: (ecg_signal, eeg_signal)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If data cannot be loaded properly
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"Loading data from {file_path}...")

    try:
        # Read CSV file, skipping OpenBCI header (first 5 lines)
        data = pd.read_csv(file_path, skiprows=5, header=None, nrows=max_samples)

        # Extract ECG (column 1) and EEG (column 2) data
        ecg = data.iloc[:, 1].values.astype(float)
        eeg = data.iloc[:, 2].values.astype(float)

        # Remove NaN values
        valid_indices = ~(np.isnan(ecg) | np.isnan(eeg))
        ecg = ecg[valid_indices]
        eeg = eeg[valid_indices]

        print(f"  Loaded {len(ecg)} valid samples")
        return ecg, eeg

    except Exception as e:
        raise ValueError(f"Error loading data from {file_path}: {str(e)}")


def process_single_file(file_path: str, max_samples: Optional[int] = 50000,
                       eeg_lowcut: float = 1.0, eeg_highcut: float = 35.0,
                       sampling_rate: float = 250.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and process a single OpenBCI file with EEG filtering and R-peak detection.

    Args:
        file_path (str): Path to the data file
        max_samples (int, optional): Maximum samples to load
        eeg_lowcut (float): EEG bandpass filter low cutoff
        eeg_highcut (float): EEG bandpass filter high cutoff
        sampling_rate (float): Signal sampling rate

    Returns:
        tuple: (eeg_filtered, rpeak_binary, ecg_cleaned, rpeak_locations)
    """
    # Load raw data
    ecg, eeg = load_openbci_file(file_path, max_samples)

    # Apply bandpass filter to EEG
    print(f"  Applying bandpass filter to EEG ({eeg_lowcut}-{eeg_highcut} Hz)...")
    eeg_filtered = apply_bandpass_filter(eeg, eeg_lowcut, eeg_highcut, sampling_rate)

    # Detect R-peaks from ECG
    print("  Detecting R-peaks from ECG...")
    ecg_cleaned, rpeak_binary, rpeak_locations = detect_r_peaks(ecg, sampling_rate)

    print(f"  Found {len(rpeak_locations)} R-peaks")

    return eeg_filtered, rpeak_binary, ecg_cleaned, rpeak_locations


def load_all_files(file_pattern: str = "*.txt", max_samples: Optional[int] = 50000,
                  **processing_kwargs) -> Tuple[List[np.ndarray], List[np.ndarray],
                                                List[np.ndarray], List[np.ndarray]]:
    """
    Load and process all files matching the given pattern.

    Args:
        file_pattern (str): Glob pattern to match files
        max_samples (int, optional): Maximum samples per file
        **processing_kwargs: Additional arguments for process_single_file

    Returns:
        tuple: (all_eeg_filtered, all_rpeak_binary, all_ecg_cleaned, all_rpeak_locations)
    """
    files = glob.glob(file_pattern)

    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")

    print(f"Found {len(files)} files matching pattern '{file_pattern}'")

    all_eeg_filtered = []
    all_rpeak_binary = []
    all_ecg_cleaned = []
    all_rpeak_locations = []

    for file_path in files:
        eeg_filtered, rpeak_binary, ecg_cleaned, rpeak_locations = process_single_file(
            file_path, max_samples, **processing_kwargs
        )

        all_eeg_filtered.append(eeg_filtered)
        all_rpeak_binary.append(rpeak_binary)
        all_ecg_cleaned.append(ecg_cleaned)
        all_rpeak_locations.append(rpeak_locations)

    print(f"Successfully processed all {len(files)} files")

    return all_eeg_filtered, all_rpeak_binary, all_ecg_cleaned, all_rpeak_locations


def get_data_statistics(eeg_data: List[np.ndarray], rpeak_data: List[np.ndarray]) -> dict:
    """
    Calculate statistics for the loaded dataset.

    Args:
        eeg_data (List[np.ndarray]): List of EEG signal arrays
        rpeak_data (List[np.ndarray]): List of R-peak binary arrays

    Returns:
        dict: Dataset statistics
    """
    total_samples = sum(len(eeg) for eeg in eeg_data)
    total_rpeaks = sum(np.sum(rpeak) for rpeak in rpeak_data)
    rpeak_percentage = (total_rpeaks / total_samples) * 100

    stats = {
        'total_files': len(eeg_data),
        'total_samples': total_samples,
        'total_rpeaks': int(total_rpeaks),
        'rpeak_percentage': rpeak_percentage,
        'imbalance_ratio': total_samples / total_rpeaks if total_rpeaks > 0 else float('inf'),
        'avg_samples_per_file': total_samples / len(eeg_data),
        'avg_rpeaks_per_file': total_rpeaks / len(eeg_data)
    }

    return stats
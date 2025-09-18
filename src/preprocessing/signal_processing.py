"""
Signal processing utilities for ECG and EEG data.
"""

import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy import signal
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def apply_bandpass_filter(eeg_signal, lowcut=1.0, highcut=35.0, fs=250.0, order=4):
    """
    Apply bandpass filter to EEG signal.

    Args:
        eeg_signal (np.ndarray): EEG signal to filter
        lowcut (float): Low cutoff frequency in Hz
        highcut (float): High cutoff frequency in Hz
        fs (float): Sampling frequency in Hz
        order (int): Filter order

    Returns:
        np.ndarray: Filtered EEG signal
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_eeg = signal.filtfilt(b, a, eeg_signal)
    return filtered_eeg


def clean_ecg_signal(ecg_signal, sampling_rate=250):
    """
    Clean ECG signal using neurokit2.

    Args:
        ecg_signal (np.ndarray): Raw ECG signal
        sampling_rate (int): Sampling frequency in Hz

    Returns:
        np.ndarray: Cleaned ECG signal
    """
    return nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)


def detect_r_peaks(ecg_signal, sampling_rate=250):
    """
    Detect R-peaks from ECG signal.

    Args:
        ecg_signal (np.ndarray): ECG signal (preferably cleaned)
        sampling_rate (int): Sampling frequency in Hz

    Returns:
        tuple: (cleaned_ecg, rpeak_binary, rpeak_locations)
            - cleaned_ecg: Cleaned ECG signal
            - rpeak_binary: Binary signal with 1s at R-peak locations
            - rpeak_locations: Array of R-peak sample indices
    """
    # Clean ECG if not already cleaned
    ecg_cleaned = clean_ecg_signal(ecg_signal, sampling_rate)

    # Detect R-peaks
    _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
    rpeak_locations = rpeaks['ECG_R_Peaks']

    # Create binary signal
    rpeak_binary = np.zeros(len(ecg_signal))
    rpeak_binary[rpeak_locations] = 1

    return ecg_cleaned, rpeak_binary, rpeak_locations


def standardize_signal(signal, scaler=None, fit=False):
    """
    Standardize signal using z-score normalization.

    Args:
        signal (np.ndarray): Signal to standardize
        scaler (StandardScaler, optional): Pre-fitted scaler
        fit (bool): Whether to fit the scaler on this signal

    Returns:
        tuple: (standardized_signal, scaler)
    """
    signal_reshaped = signal.reshape(-1, 1)

    if scaler is None:
        scaler = StandardScaler()

    if fit:
        standardized = scaler.fit_transform(signal_reshaped).flatten()
    else:
        standardized = scaler.transform(signal_reshaped).flatten()

    return standardized, scaler


def prepare_data_splits(eeg_data, rpeak_data, train_ratio=0.65, val_ratio=0.15):
    """
    Prepare time-based data splits for training, validation, and testing.

    Args:
        eeg_data (list): List of EEG signal arrays
        rpeak_data (list): List of R-peak binary arrays
        train_ratio (float): Proportion for training set
        val_ratio (float): Proportion for validation set

    Returns:
        tuple: (eeg_train, eeg_val, eeg_test, rpeaks_train, rpeaks_val, rpeaks_test, scaler)
    """
    # Combine all data
    eeg_combined = np.concatenate(eeg_data)
    rpeaks_combined = np.concatenate(rpeak_data)

    # Calculate split indices
    train_split = int(len(eeg_combined) * train_ratio)
    val_split = int(len(eeg_combined) * (train_ratio + val_ratio))

    # Time-based splits
    eeg_train = eeg_combined[:train_split]
    eeg_val = eeg_combined[train_split:val_split]
    eeg_test = eeg_combined[val_split:]

    rpeaks_train = rpeaks_combined[:train_split]
    rpeaks_val = rpeaks_combined[train_split:val_split]
    rpeaks_test = rpeaks_combined[val_split:]

    # Standardize using training data statistics
    eeg_train, scaler = standardize_signal(eeg_train, fit=True)
    eeg_val, _ = standardize_signal(eeg_val, scaler, fit=False)
    eeg_test, _ = standardize_signal(eeg_test, scaler, fit=False)

    return eeg_train, eeg_val, eeg_test, rpeaks_train, rpeaks_val, rpeaks_test, scaler
#!/usr/bin/env python3
"""
IMPROVED R-PEAK DETECTION METHOD - Strategy 5: Optimized CNN Approach
====================================================================

This improved strategy implements:
1. 60Hz notch filter for power line noise removal
2. 0.5-45Hz bandpass filter for better EEG preprocessing
3. Simple CNN model instead of complex TimesNet
4. 0.5-second windows (63 samples) for single R-peak focus
5. Dual-channel input: raw + standardized EEG signals

Key Improvements:
- Shorter windows reduce complexity and increase dataset size
- Dual channels provide both raw temporal patterns and normalized features
- Simple CNN avoids overfitting compared to complex architectures
- Better signal preprocessing removes artifacts

Usage:
    python improved_rpeak_detection.py

Author: Claude Code
Date: September 19, 2025
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import neurokit2 as nk
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy import signal
import glob
import time
import warnings
warnings.filterwarnings('ignore')
import json
import datetime
import csv

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling extreme class imbalance
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        # Apply alpha weighting for class imbalance
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Apply focal weight for hard examples
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class SimpleRPeakCNN(nn.Module):
    """
    Simple 1D CNN for R-peak detection with dual-channel input

    NEW IMPROVEMENTS:
    - Designed for 0.5-second windows (63 samples)
    - Dual-channel input: raw + standardized EEG
    - Simpler architecture to prevent overfitting
    """
    def __init__(self, input_size=63, input_channels=2):
        super(SimpleRPeakCNN, self).__init__()
        self.input_size = input_size

        # Convolutional layers
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)

        # Pooling and regularization
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(32)

        # Calculate output size after convolutions and pooling
        # After conv1 + pool: 63 -> 31
        # After conv2 + pool: 31 -> 15
        # After conv3: 15 -> 15
        self.fc_input_size = 32 * 15  # 32 channels * 15 time points

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        # x shape: (batch_size, 2, 63) - dual channel input

        # First conv block
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = self.pool(x)  # (batch_size, 32, 31)

        # Second conv block
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.pool(x)  # (batch_size, 64, 15)

        # Third conv block (no pooling)
        x = F.relu(self.batch_norm3(self.conv3(x)))  # (batch_size, 32, 15)

        # Flatten and fully connected
        x = x.view(x.size(0), -1)  # (batch_size, 32*15)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class DualChannelDataset(Dataset):
    """
    Dataset with dual-channel input: raw + standardized EEG signals
    """
    def __init__(self, positive_samples, negative_samples):
        self.samples = []

        # Add positive samples (label 1)
        for raw_window, std_window in positive_samples:
            dual_channel = np.stack([raw_window.flatten(), std_window.flatten()], axis=0)
            self.samples.append((dual_channel, 1))

        # Add negative samples (label 0)
        for raw_window, std_window in negative_samples:
            dual_channel = np.stack([raw_window.flatten(), std_window.flatten()], axis=0)
            self.samples.append((dual_channel, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dual_channel, label = self.samples[idx]
        return torch.FloatTensor(dual_channel), torch.LongTensor([label])

class DualChannelValidationDataset(Dataset):
    """
    Validation dataset with dual-channel input for 0.5-second windows
    """
    def __init__(self, eeg_raw, eeg_std, rpeak_binary, window_size=63, stride=5):
        self.eeg_raw = eeg_raw
        self.eeg_std = eeg_std
        self.rpeak_binary = rpeak_binary
        self.window_size = window_size
        self.stride = stride

        # Calculate number of windows with stride
        self.num_windows = (len(eeg_raw) - window_size) // stride + 1

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size

        raw_window = self.eeg_raw[start_idx:end_idx]
        std_window = self.eeg_std[start_idx:end_idx]
        dual_channel = np.stack([raw_window, std_window], axis=0)

        # Label based on center±1 approach (tighter for 0.5s windows)
        center_idx = start_idx + self.window_size // 2
        label = 0
        for offset in [-1, 0, 1]:
            check_idx = center_idx + offset
            if 0 <= check_idx < len(self.rpeak_binary) and self.rpeak_binary[check_idx] == 1:
                label = 1
                break

        return torch.FloatTensor(dual_channel), torch.LongTensor([label]), center_idx

def apply_notch_filter(signal_data, notch_freq=60.0, fs=250.0, quality_factor=30.0):
    """
    Apply notch filter to remove power line interference

    NEW IMPROVEMENT: 60Hz notch filter for power line noise removal
    """
    nyquist = 0.5 * fs
    low = (notch_freq - notch_freq/quality_factor) / nyquist
    high = (notch_freq + notch_freq/quality_factor) / nyquist

    b, a = signal.butter(2, [low, high], btype='bandstop')
    filtered_signal = signal.filtfilt(b, a, signal_data)
    return filtered_signal

def apply_bandpass_filter(eeg_signal, lowcut=0.5, highcut=45.0, fs=250.0, order=4):
    """
    Apply bandpass filter to EEG signal

    NEW IMPROVEMENT: 0.5-45Hz bandpass filter for better EEG preprocessing
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_eeg = signal.filtfilt(b, a, eeg_signal)
    return filtered_eeg

def load_and_process_file_improved(file_path, max_samples=None):
    """
    Load and process OpenBCI EEG/ECG data file with improved preprocessing

    NEW IMPROVEMENTS:
    1. 60Hz notch filter
    2. 0.5-45Hz bandpass filter
    3. Dual-channel output: raw + standardized
    """
    print(f"Processing {file_path} with improved preprocessing...")

    data = pd.read_csv(file_path, skiprows=5, header=None, nrows=max_samples)
    ecg = data.iloc[:, 0].values.astype(float)  # ECG is channel 0
    eeg = data.iloc[:, 1].values.astype(float)  # EEG is channel 1

    # Remove NaN values
    valid_indices = ~(np.isnan(ecg) | np.isnan(eeg))
    ecg = ecg[valid_indices]
    eeg = eeg[valid_indices]

    print(f"  Loaded {len(ecg)} samples at 250Hz")

    # Step 1: Apply 60Hz notch filter to remove power line noise
    print("  Applying 60Hz notch filter...")
    eeg_notched = apply_notch_filter(eeg, notch_freq=60.0, fs=250.0)

    # Step 2: Apply improved bandpass filter (0.5-45Hz)
    print("  Applying 0.5-45Hz bandpass filter...")
    eeg_filtered = apply_bandpass_filter(eeg_notched, lowcut=0.5, highcut=45.0, fs=250.0)

    # Detect R-peaks from ECG using neurokit2 at original sampling rate
    ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=250)
    _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=250)

    # Create binary R-peak signal
    rpeak_binary = np.zeros(len(ecg))
    rpeak_locations = rpeaks['ECG_R_Peaks']
    rpeak_binary[rpeak_locations] = 1

    print(f"  Found {len(rpeak_locations)} R-peaks at 250Hz")

    # Downsample to 125Hz by taking every other sample
    eeg_raw_downsampled = eeg_filtered[::2]
    ecg_downsampled = ecg_cleaned[::2]

    # Step 3: Create standardized version for dual-channel input
    print("  Creating standardized signal for dual-channel input...")
    scaler = StandardScaler()
    eeg_std_downsampled = scaler.fit_transform(eeg_raw_downsampled.reshape(-1, 1)).flatten()

    # Properly downsample R-peak binary signal
    rpeak_binary_downsampled = np.zeros(len(eeg_raw_downsampled))
    rpeak_locations_downsampled = rpeak_locations // 2  # Adjust indices for 125Hz

    # Set R-peaks in downsampled binary signal
    for rpeak_idx in rpeak_locations_downsampled:
        if 0 <= rpeak_idx < len(rpeak_binary_downsampled):
            rpeak_binary_downsampled[rpeak_idx] = 1

    print(f"  Downsampled to {len(eeg_raw_downsampled)} samples at 125Hz")
    print(f"  R-peaks at 125Hz: {np.sum(rpeak_binary_downsampled)}")

    return (eeg_raw_downsampled, eeg_std_downsampled, rpeak_binary_downsampled,
            ecg_downsampled, rpeak_locations_downsampled, scaler)

def create_dual_channel_training_samples(eeg_raw, eeg_std, rpeak_binary, window_size=63, negative_ratio=10):
    """
    Create training samples with 0.5-second windows and dual-channel input

    NEW IMPROVEMENTS:
    - 0.5-second windows (63 samples at 125Hz)
    - Dual-channel: raw + standardized signals
    """
    print(f"Creating dual-channel training samples with 0.5-second windows ({window_size} samples)...")

    positive_samples = []
    negative_candidates = []

    # IMPORTANT: stride=1 sampling strategy
    # This creates overlapping windows to maximize training data
    # Example: for 63-sample windows, we get N-62 total windows where N is signal length
    for start_idx in range(len(eeg_raw) - window_size + 1):
        center_idx = start_idx + window_size // 2

        # LABELING STRATEGY: Check if this window should be positive
        # Positive: R-peak within center±1 samples (±8ms tolerance at 125Hz)
        # This tight tolerance works better for 0.5s windows vs 3s windows
        is_positive = False
        for offset in [-1, 0, 1]:  # Tighter tolerance for 0.5s windows
            check_idx = center_idx + offset
            if 0 <= check_idx < len(rpeak_binary) and rpeak_binary[check_idx] == 1:
                is_positive = True
                break

        raw_window = eeg_raw[start_idx:start_idx + window_size].reshape(-1, 1)
        std_window = eeg_std[start_idx:start_idx + window_size].reshape(-1, 1)

        if is_positive:
            positive_samples.append((raw_window, std_window))
        else:
            # Realistic negative mining - at least 3 samples (24ms at 125Hz) away
            min_distance_to_rpeak = float('inf')
            for rpeak_idx in np.where(rpeak_binary == 1)[0]:
                distance = abs(center_idx - rpeak_idx)
                min_distance_to_rpeak = min(min_distance_to_rpeak, distance)

            if min_distance_to_rpeak >= 3:  # 24ms away at 125Hz
                negative_candidates.append((raw_window, std_window))

    # Use specified negative:positive ratio
    num_positive = len(positive_samples)
    target_negatives = num_positive * negative_ratio

    if len(negative_candidates) >= target_negatives:
        negative_indices = np.random.choice(len(negative_candidates), target_negatives, replace=False)
        negative_samples = [negative_candidates[i] for i in negative_indices]
    else:
        negative_samples = negative_candidates
        print(f"Warning: Only {len(negative_candidates)} negative candidates available, wanted {target_negatives}")

    print(f"Created {len(positive_samples)} positive and {len(negative_samples)} negative samples")

    total_samples = len(positive_samples) + len(negative_samples)
    if total_samples > 0:
        print(f"Training ratio - Positive: {len(positive_samples)/total_samples*100:.1f}%, Negative: {len(negative_samples)/total_samples*100:.1f}%")
    else:
        print("ERROR: No training samples created!")

    return positive_samples, negative_samples

def prepare_improved_data(all_eeg_raw, all_eeg_std, all_rpeaks, train_ratio=0.7):
    """Prepare improved dual-channel training data"""
    print("Preparing improved dual-channel training data...")

    eeg_raw_combined = np.concatenate(all_eeg_raw)
    eeg_std_combined = np.concatenate(all_eeg_std)
    rpeaks_combined = np.concatenate(all_rpeaks)

    train_split = int(len(eeg_raw_combined) * train_ratio)

    # Split data chronologically (temporal split)
    eeg_raw_train = eeg_raw_combined[:train_split]
    eeg_raw_val = eeg_raw_combined[train_split:]
    eeg_std_train = eeg_std_combined[:train_split]
    eeg_std_val = eeg_std_combined[train_split:]
    rpeaks_train = rpeaks_combined[:train_split]
    rpeaks_val = rpeaks_combined[train_split:]

    print(f"Train: {len(eeg_raw_train)} samples, R-peaks: {np.sum(rpeaks_train)}")
    print(f"Validation: {len(eeg_raw_val)} samples, R-peaks: {np.sum(rpeaks_val)}")

    # Create dual-channel training samples
    train_positive, train_negative = create_dual_channel_training_samples(
        eeg_raw_train, eeg_std_train, rpeaks_train
    )

    return (train_positive, train_negative, eeg_raw_val, eeg_std_val, rpeaks_val)

def train_improved_cnn(train_positive, train_negative, eeg_raw_val, eeg_std_val, rpeaks_val,
                      epochs=25, batch_size=64):
    """
    Train improved CNN with all enhancements
    """
    print("Training IMPROVED METHOD: Simple CNN with dual-channel input...")

    # Setup logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = 'outputs/logs'
    os.makedirs(log_dir, exist_ok=True)

    # Log files
    epoch_log_file = os.path.join(log_dir, f'improved_training_log_{timestamp}.csv')
    config_log_file = os.path.join(log_dir, f'improved_config_{timestamp}.json')

    # Initialize CSV log for epoch-wise data
    with open(epoch_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_precision', 'train_recall', 'train_f1', 'train_accuracy',
                        'val_loss', 'val_precision', 'val_recall', 'val_f1', 'val_accuracy',
                        'learning_rate', 'epoch_time'])

    # Create dual-channel training dataset
    train_dataset = DualChannelDataset(train_positive, train_negative)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Create dual-channel validation dataset
    # IMPORTANT: Validation uses stride=5 (40ms spacing) for efficiency
    # This reduces computational load while still providing good coverage
    val_dataset = DualChannelValidationDataset(eeg_raw_val, eeg_std_val, rpeaks_val, stride=5)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize improved CNN model
    model = SimpleRPeakCNN(input_size=63, input_channels=2).to(device)

    # Calculate class weights
    num_positive = len(train_positive)
    num_negative = len(train_negative)
    total_samples = num_positive + num_negative

    weight_negative = total_samples / (2 * num_negative)
    weight_positive = total_samples / (2 * num_positive)
    class_weights = torch.tensor([weight_negative, weight_positive], dtype=torch.float32).to(device)

    print(f"Class weights - Negative: {weight_negative:.4f}, Positive: {weight_positive:.4f}")

    # Use CrossEntropyLoss with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    # Log configuration
    config_data = {
        'timestamp': timestamp,
        'model_config': {
            'architecture': 'SimpleRPeakCNN',
            'input_size': 63,
            'input_channels': 2,
            'window_size_seconds': 0.5,
            'sampling_rate': 125
        },
        'training_config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'optimizer': 'AdamW',
            'learning_rate': 0.001,
            'weight_decay': 1e-3
        },
        'preprocessing': {
            'notch_filter': '60Hz',
            'bandpass_filter': '0.5-45Hz',
            'dual_channel': 'raw + standardized'
        },
        'data_config': {
            'train_positive_samples': len(train_positive),
            'train_negative_samples': len(train_negative),
            'validation_samples': len(eeg_raw_val),
            'validation_rpeaks': np.sum(rpeaks_val)
        },
        'device': str(device)
    }

    with open(config_log_file, 'w') as f:
        json.dump(config_data, f, indent=2)

    print(f"Logging configuration saved to: {config_log_file}")

    # Training tracking
    best_f1 = 0.0
    patience_counter = 0

    def calculate_metrics(scores, targets, threshold=0.5):
        binary_preds = (scores >= threshold).astype(int)

        if len(np.unique(targets)) < 2:
            return 0.0, 0.0, 0.0, 0.0

        precision, recall, f1, _ = precision_recall_fscore_support(targets, binary_preds, average='binary', zero_division=0)
        accuracy = np.mean(binary_preds == targets)

        return precision, recall, f1, accuracy

    print("\n--- Training Progress ---")
    for epoch in range(epochs):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_scores, train_targets = [], []

        for data, target in train_loader:
            data, target = data.to(device), target.to(device).squeeze()

            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred_probs = torch.softmax(logits, dim=1)[:, 1]

            train_scores.extend(pred_probs.detach().cpu().numpy())
            train_targets.extend(target.cpu().numpy())

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_scores, val_targets = [], []

        with torch.no_grad():
            for data, target, _ in val_loader:
                data, target = data.to(device), target.to(device).squeeze()

                logits = model(data)
                loss = criterion(logits, target)
                val_loss += loss.item()

                pred_probs = torch.softmax(logits, dim=1)[:, 1]

                val_scores.extend(pred_probs.cpu().numpy())
                val_targets.extend(target.cpu().numpy())

        # Calculate metrics
        train_precision, train_recall, train_f1, train_accuracy = calculate_metrics(
            np.array(train_scores), np.array(train_targets)
        )
        val_precision, val_recall, val_f1, val_accuracy = calculate_metrics(
            np.array(val_scores), np.array(val_targets)
        )

        scheduler.step()

        # Calculate epoch time and current learning rate
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']

        # Log epoch data
        with open(epoch_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss/len(train_loader), train_precision, train_recall, train_f1, train_accuracy,
                           val_loss/len(val_loader), val_precision, val_recall, val_f1, val_accuracy,
                           current_lr, epoch_time])

        # Early stopping based on F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            os.makedirs('outputs/models', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': config_data['model_config']
            }, 'outputs/models/improved_rpeak_model.pth')
        else:
            patience_counter += 1

        print(f'Epoch {epoch:2d}: Loss: {val_loss/len(val_loader):.4f}, '
              f'F1: {val_f1:.3f}, Precision: {val_precision:.3f}, '
              f'Recall: {val_recall:.3f}, Time: {epoch_time:.1f}s')

        if patience_counter >= 7:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    if os.path.exists('outputs/models/improved_rpeak_model.pth'):
        try:
            checkpoint = torch.load('outputs/models/improved_rpeak_model.pth', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"\nLoaded best model")
        except Exception as e:
            print(f"\nCould not load checkpoint: {e}")

    return model

def evaluate_improved_model(model, eeg_raw_val, eeg_std_val, rpeaks_val):
    """Evaluate the improved model"""
    print("\n--- Final Evaluation ---")
    print("Evaluating IMPROVED METHOD on validation set...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    # Create validation dataset
    val_dataset = DualChannelValidationDataset(eeg_raw_val, eeg_std_val, rpeaks_val, stride=5)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)

    prediction_scores = []
    true_labels = []

    with torch.no_grad():
        for data, target, _ in val_loader:
            data = data.to(device)

            logits = model(data)
            pred_probs = torch.softmax(logits, dim=1)[:, 1]

            prediction_scores.extend(pred_probs.cpu().numpy())
            true_labels.extend(target.numpy().flatten())

    prediction_scores = np.array(prediction_scores)
    true_labels = np.array(true_labels)
    predictions_thresh = (prediction_scores >= 0.5).astype(int)

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions_thresh, average='binary', zero_division=0)
    accuracy = np.mean(predictions_thresh == true_labels)
    cm = confusion_matrix(true_labels, predictions_thresh)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\n🏆 IMPROVED METHOD RESULTS:")
    print(f"Accuracy:     {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"Precision:    {precision:.4f} ({precision*100:.1f}%)")
    print(f"Recall:       {recall:.4f} ({recall*100:.1f}%)")
    print(f"Specificity:  {specificity:.4f} ({specificity*100:.1f}%)")
    print(f"F1-Score:     {f1:.4f} ({f1*100:.1f}%)")
    print(f"\n📊 CONFUSION MATRIX:")
    print(f"True Positives (TP):  {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN):  {tn}")
    print(f"False Negatives (FN): {fn}")

    results = {
        'strategy': 'IMPROVED METHOD: Simple CNN with dual-channel input',
        'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'specificity': specificity, 'f1': f1,
        'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
    }

    return results

def main():
    """Main execution function - IMPROVED R-PEAK DETECTION METHOD"""
    print("🚀 ===== IMPROVED R-PEAK DETECTION METHOD =====")
    print("Strategy 5: Simple CNN with Enhanced Preprocessing")
    print("Improvements: 60Hz notch + 0.5-45Hz bandpass + 0.5s windows + dual-channel")
    print("================================================\n")

    # Use specific OpenBCI data file
    target_file = "OpenBCI-RAW-2025-09-14_12-26-20.txt"

    if not os.path.exists(target_file):
        print(f"ERROR: Target file {target_file} not found!")
        print("Please ensure the file is in the current directory")
        return None

    print(f"Using single file: {target_file}")
    all_eeg_raw = []
    all_eeg_std = []
    all_rpeaks = []

    # Process single data file with improved preprocessing
    (eeg_raw, eeg_std, rpeak_binary, ecg_cleaned,
     rpeak_locations, scaler) = load_and_process_file_improved(target_file)

    all_eeg_raw.append(eeg_raw)
    all_eeg_std.append(eeg_std)
    all_rpeaks.append(rpeak_binary)

    # Prepare improved dual-channel data
    (train_positive, train_negative, eeg_raw_val,
     eeg_std_val, rpeaks_val) = prepare_improved_data(all_eeg_raw, all_eeg_std, all_rpeaks)

    # Train improved model
    model = train_improved_cnn(train_positive, train_negative, eeg_raw_val, eeg_std_val, rpeaks_val,
                              epochs=20, batch_size=64)

    # Final evaluation
    results = evaluate_improved_model(model, eeg_raw_val, eeg_std_val, rpeaks_val)

    # Generate training curves visualization
    print(f"\n📊 Generating training curves visualization...")
    try:
        log_files = sorted([f for f in os.listdir('outputs/logs') if f.startswith('improved_training_log_') and f.endswith('.csv')])
        if log_files:
            latest_log = os.path.join('outputs/logs', log_files[-1])
            df = pd.read_csv(latest_log)

            # Create training curves plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

            epochs = df['epoch']

            # Loss curves
            ax1.plot(epochs, df['train_loss'], 'b-', label='Training Loss', linewidth=2)
            ax1.plot(epochs, df['val_loss'], 'r-', label='Validation Loss', linewidth=2)
            ax1.set_title('Loss Curves')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)

            # F1 Score
            ax2.plot(epochs, df['train_f1'], 'b-', label='Training F1', linewidth=2)
            ax2.plot(epochs, df['val_f1'], 'r-', label='Validation F1', linewidth=2)
            ax2.set_title('F1 Score Curves')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('F1 Score')
            ax2.legend()
            ax2.grid(True)

            # Precision curves
            ax3.plot(epochs, df['train_precision'], 'b-', label='Training Precision', linewidth=2)
            ax3.plot(epochs, df['val_precision'], 'r-', label='Validation Precision', linewidth=2)
            ax3.set_title('Precision Curves')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Precision')
            ax3.legend()
            ax3.grid(True)

            # Recall curves
            ax4.plot(epochs, df['train_recall'], 'b-', label='Training Recall', linewidth=2)
            ax4.plot(epochs, df['val_recall'], 'r-', label='Validation Recall', linewidth=2)
            ax4.set_title('Recall Curves')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Recall')
            ax4.legend()
            ax4.grid(True)

            plt.tight_layout()
            plt.savefig('outputs/plots/improved_training_curves.png', dpi=300, bbox_inches='tight')
            print(f"📈 Training curves saved: outputs/plots/improved_training_curves.png")
            plt.close()
    except Exception as e:
        print(f"⚠️ Could not generate visualization: {e}")

    print(f"\n🎉 SUCCESS: Improved method completed!")
    print(f"📁 Model saved: outputs/models/improved_rpeak_model.pth")

    return results

if __name__ == "__main__":
    result = main()
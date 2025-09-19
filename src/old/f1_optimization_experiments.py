#!/usr/bin/env python3
"""
F1-SCORE OPTIMIZATION EXPERIMENTS
=================================

Systematic experimentation framework to maximize F1-score for R-peak detection.
This script runs 40+ different experimental configurations across multiple dimensions:

1. Signal Preprocessing Variations (10+ trials)
2. Data Strategy Variations (10+ trials)
3. Model Architecture Variations (10+ trials)
4. Training Strategy Variations (10+ trials)
5. Post-processing Techniques (5+ trials)

Each experiment is tracked with parameters and results for systematic analysis.

Usage:
    source /home/heonsoo/venvs/np1/bin/activate
    python src/f1_optimization_experiments.py

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
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from scipy import signal
import glob
import time
import warnings
warnings.filterwarnings('ignore')
import json
import datetime
import csv
from itertools import product
import copy

class ExperimentTracker:
    """Track all experiments and their results"""
    def __init__(self):
        self.results = []
        self.best_f1 = 0.0
        self.best_config = None

    def log_experiment(self, exp_id, config, results):
        """Log an experiment result"""
        entry = {
            'exp_id': exp_id,
            'config': copy.deepcopy(config),
            'precision': results.get('precision', 0),
            'recall': results.get('recall', 0),
            'f1': results.get('f1', 0),
            'accuracy': results.get('accuracy', 0),
            'specificity': results.get('specificity', 0),
            'tp': results.get('tp', 0),
            'fp': results.get('fp', 0),
            'tn': results.get('tn', 0),
            'fn': results.get('fn', 0),
            'training_time': results.get('training_time', 0)
        }

        self.results.append(entry)

        if entry['f1'] > self.best_f1:
            self.best_f1 = entry['f1']
            self.best_config = copy.deepcopy(config)
            print(f"🏆 NEW BEST F1: {self.best_f1:.4f} (Exp {exp_id})")

        print(f"Exp {exp_id:2d}: F1={entry['f1']:.4f}, P={entry['precision']:.4f}, R={entry['recall']:.4f}")

    def save_results(self, filename='outputs/f1_optimization_results.csv'):
        """Save all results to CSV"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Flatten config for CSV
        flattened_results = []
        for result in self.results:
            flat = {
                'exp_id': result['exp_id'],
                'precision': result['precision'],
                'recall': result['recall'],
                'f1': result['f1'],
                'accuracy': result['accuracy'],
                'specificity': result['specificity'],
                'tp': result['tp'],
                'fp': result['fp'],
                'tn': result['tn'],
                'fn': result['fn'],
                'training_time': result['training_time']
            }

            # Flatten config dict
            config = result['config']
            for key, value in config.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        flat[f'{key}_{subkey}'] = subvalue
                else:
                    flat[key] = value

            flattened_results.append(flat)

        df = pd.DataFrame(flattened_results)
        df.to_csv(filename, index=False)
        print(f"📊 Results saved to {filename}")

    def print_top_results(self, top_k=10):
        """Print top K results"""
        sorted_results = sorted(self.results, key=lambda x: x['f1'], reverse=True)

        print(f"\n🏆 TOP {top_k} EXPERIMENTS BY F1-SCORE:")
        print("="*80)
        for i, result in enumerate(sorted_results[:top_k]):
            print(f"{i+1:2d}. Exp {result['exp_id']:2d}: F1={result['f1']:.4f}, "
                  f"P={result['precision']:.4f}, R={result['recall']:.4f}")

            # Print key config differences
            config = result['config']
            key_params = []
            if 'preprocessing' in config:
                if 'bandpass_low' in config['preprocessing']:
                    key_params.append(f"BP:{config['preprocessing']['bandpass_low']}-{config['preprocessing']['bandpass_high']}Hz")
                if 'notch_freqs' in config['preprocessing']:
                    key_params.append(f"Notch:{config['preprocessing']['notch_freqs']}")
            if 'data' in config:
                if 'window_size' in config['data']:
                    key_params.append(f"Win:{config['data']['window_size']}")
                if 'negative_ratio' in config['data']:
                    key_params.append(f"Ratio:1:{config['data']['negative_ratio']}")
            if 'model' in config:
                if 'architecture' in config['model']:
                    key_params.append(f"Arch:{config['model']['architecture']}")
            if 'training' in config:
                if 'loss_function' in config['training']:
                    key_params.append(f"Loss:{config['training']['loss_function']}")

            print(f"    {' | '.join(key_params)}")
        print("="*80)

class ConfigurableDataProcessor:
    """Flexible data processor with configurable preprocessing"""

    @staticmethod
    def apply_notch_filter(signal_data, notch_freqs=[60], fs=250.0, quality_factor=30.0):
        """Apply multiple notch filters"""
        filtered_signal = signal_data.copy()
        for notch_freq in notch_freqs:
            nyquist = 0.5 * fs
            low = (notch_freq - notch_freq/quality_factor) / nyquist
            high = (notch_freq + notch_freq/quality_factor) / nyquist
            b, a = signal.butter(2, [low, high], btype='bandstop')
            filtered_signal = signal.filtfilt(b, a, filtered_signal)
        return filtered_signal

    @staticmethod
    def apply_bandpass_filter(eeg_signal, lowcut=0.5, highcut=45.0, fs=250.0, order=4, filter_type='butter'):
        """Apply configurable bandpass filter"""
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist

        if filter_type == 'butter':
            b, a = signal.butter(order, [low, high], btype='band')
        elif filter_type == 'cheby1':
            b, a = signal.cheby1(order, 0.5, [low, high], btype='band')
        elif filter_type == 'ellip':
            b, a = signal.ellip(order, 0.5, 40, [low, high], btype='band')
        else:
            b, a = signal.butter(order, [low, high], btype='band')

        filtered_eeg = signal.filtfilt(b, a, eeg_signal)
        return filtered_eeg

    @staticmethod
    def apply_baseline_correction(signal_data, method='linear_detrend'):
        """Apply baseline correction"""
        if method == 'linear_detrend':
            return signal.detrend(signal_data, type='linear')
        elif method == 'mean_removal':
            return signal_data - np.mean(signal_data)
        elif method == 'median_removal':
            return signal_data - np.median(signal_data)
        else:
            return signal_data

    @staticmethod
    def apply_normalization(signal_data, method='standard'):
        """Apply different normalization methods"""
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'none':
            return signal_data, None
        else:
            scaler = StandardScaler()

        normalized = scaler.fit_transform(signal_data.reshape(-1, 1)).flatten()
        return normalized, scaler

    @staticmethod
    def load_and_process_configurable(file_path, preprocessing_config):
        """Load and process with configurable preprocessing"""
        print(f"Processing {file_path} with config: {preprocessing_config}")

        # Load data
        data = pd.read_csv(file_path, skiprows=5, header=None)
        ecg = data.iloc[:, 0].values.astype(float)
        eeg = data.iloc[:, 1].values.astype(float)

        # Remove NaN values
        valid_indices = ~(np.isnan(ecg) | np.isnan(eeg))
        ecg = ecg[valid_indices]
        eeg = eeg[valid_indices]

        # Apply baseline correction
        if preprocessing_config.get('baseline_correction', 'none') != 'none':
            eeg = ConfigurableDataProcessor.apply_baseline_correction(
                eeg, preprocessing_config['baseline_correction']
            )

        # Apply notch filters
        if 'notch_freqs' in preprocessing_config:
            eeg = ConfigurableDataProcessor.apply_notch_filter(
                eeg, preprocessing_config['notch_freqs'], fs=250.0
            )

        # Apply bandpass filter
        eeg_filtered = ConfigurableDataProcessor.apply_bandpass_filter(
            eeg,
            lowcut=preprocessing_config.get('bandpass_low', 0.5),
            highcut=preprocessing_config.get('bandpass_high', 45.0),
            order=preprocessing_config.get('filter_order', 4),
            filter_type=preprocessing_config.get('filter_type', 'butter')
        )

        # Detect R-peaks from ECG
        ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=250)
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=250)

        # Create binary R-peak signal
        rpeak_binary = np.zeros(len(ecg))
        rpeak_locations = rpeaks['ECG_R_Peaks']
        rpeak_binary[rpeak_locations] = 1

        # Downsample to target sampling rate
        target_fs = preprocessing_config.get('target_sampling_rate', 125)
        downsample_factor = 250 // target_fs

        eeg_downsampled = eeg_filtered[::downsample_factor]
        ecg_downsampled = ecg_cleaned[::downsample_factor]

        # Downsample R-peak binary signal
        rpeak_binary_downsampled = np.zeros(len(eeg_downsampled))
        rpeak_locations_downsampled = rpeak_locations // downsample_factor

        for rpeak_idx in rpeak_locations_downsampled:
            if 0 <= rpeak_idx < len(rpeak_binary_downsampled):
                rpeak_binary_downsampled[rpeak_idx] = 1

        # Apply normalization
        eeg_raw = eeg_downsampled.copy()
        eeg_normalized, scaler = ConfigurableDataProcessor.apply_normalization(
            eeg_downsampled, preprocessing_config.get('normalization', 'standard')
        )

        print(f"  Processed: {len(eeg_raw)} samples at {target_fs}Hz, R-peaks: {np.sum(rpeak_binary_downsampled)}")

        return (eeg_raw, eeg_normalized, rpeak_binary_downsampled,
                ecg_downsampled, rpeak_locations_downsampled, scaler)

# Model architectures
class SimpleCNN(nn.Module):
    """Original simple CNN"""
    def __init__(self, input_size=63, input_channels=2, hidden_dim=32):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, hidden_dim, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(hidden_dim*2, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim*2)
        self.batch_norm3 = nn.BatchNorm1d(hidden_dim)
        self.fc_input_size = hidden_dim * (input_size // 4)
        self.fc1 = nn.Linear(self.fc_input_size, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class DeepCNN(nn.Module):
    """Deeper CNN with more layers"""
    def __init__(self, input_size=63, input_channels=2, hidden_dim=32):
        super(DeepCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, hidden_dim, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(hidden_dim*2, hidden_dim*2, kernel_size=5, padding=2)
        self.conv5 = nn.Conv1d(hidden_dim*2, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm3 = nn.BatchNorm1d(hidden_dim*2)
        self.batch_norm4 = nn.BatchNorm1d(hidden_dim*2)
        self.batch_norm5 = nn.BatchNorm1d(hidden_dim)
        self.fc_input_size = hidden_dim * (input_size // 4)
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = F.relu(self.batch_norm4(self.conv4(x)))
        x = self.pool(x)
        x = F.relu(self.batch_norm5(self.conv5(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class ResNetBlock(nn.Module):
    """ResNet-style block with skip connections"""
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(channels)
        self.batch_norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        residual = x
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.batch_norm2(self.conv2(x))
        x = x + residual  # Skip connection
        x = F.relu(x)
        return x

class ResNetCNN(nn.Module):
    """ResNet-style CNN with skip connections"""
    def __init__(self, input_size=63, input_channels=2, hidden_dim=32):
        super(ResNetCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, hidden_dim, kernel_size=7, padding=3)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.resblock1 = ResNetBlock(hidden_dim)
        self.resblock2 = ResNetBlock(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=5, padding=2)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim*2)
        self.resblock3 = ResNetBlock(hidden_dim*2)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.fc_input_size = hidden_dim*2 * (input_size // 4)
        self.fc1 = nn.Linear(self.fc_input_size, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = self.pool(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.pool(x)
        x = self.resblock3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Loss functions
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        return focal_loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Convert logits to probabilities
        probs = F.softmax(inputs, dim=1)[:, 1]  # Probability of positive class
        targets_f = targets.float()

        intersection = (probs * targets_f).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets_f.sum() + self.smooth)
        return 1 - dice

# Dataset classes
class ConfigurableDataset(Dataset):
    """Configurable dataset with different window strategies"""
    def __init__(self, positive_samples, negative_samples, data_config):
        self.samples = []
        self.data_config = data_config

        # Add positive samples
        for raw_window, norm_window in positive_samples:
            if data_config.get('input_channels', 2) == 2:
                dual_channel = np.stack([raw_window.flatten(), norm_window.flatten()], axis=0)
            else:
                dual_channel = raw_window.flatten().reshape(1, -1)
            self.samples.append((dual_channel, 1))

        # Add negative samples
        for raw_window, norm_window in negative_samples:
            if data_config.get('input_channels', 2) == 2:
                dual_channel = np.stack([raw_window.flatten(), norm_window.flatten()], axis=0)
            else:
                dual_channel = raw_window.flatten().reshape(1, -1)
            self.samples.append((dual_channel, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dual_channel, label = self.samples[idx]
        return torch.FloatTensor(dual_channel), torch.LongTensor([label])

def create_training_samples_configurable(eeg_raw, eeg_norm, rpeak_binary, data_config):
    """Create training samples with configurable parameters"""
    window_size = data_config.get('window_size', 63)
    negative_ratio = data_config.get('negative_ratio', 10)
    label_tolerance = data_config.get('label_tolerance', 1)
    min_negative_distance = data_config.get('min_negative_distance', 3)

    positive_samples = []
    negative_candidates = []

    # Sample with specified stride
    stride = data_config.get('training_stride', 1)
    for start_idx in range(0, len(eeg_raw) - window_size + 1, stride):
        center_idx = start_idx + window_size // 2

        # Check if positive (R-peak within tolerance)
        is_positive = False
        for offset in range(-label_tolerance, label_tolerance + 1):
            check_idx = center_idx + offset
            if 0 <= check_idx < len(rpeak_binary) and rpeak_binary[check_idx] == 1:
                is_positive = True
                break

        raw_window = eeg_raw[start_idx:start_idx + window_size].reshape(-1, 1)
        norm_window = eeg_norm[start_idx:start_idx + window_size].reshape(-1, 1)

        if is_positive:
            positive_samples.append((raw_window, norm_window))
        else:
            # Check minimum distance for negatives
            min_distance_to_rpeak = float('inf')
            for rpeak_idx in np.where(rpeak_binary == 1)[0]:
                distance = abs(center_idx - rpeak_idx)
                min_distance_to_rpeak = min(min_distance_to_rpeak, distance)

            if min_distance_to_rpeak >= min_negative_distance:
                negative_candidates.append((raw_window, norm_window))

    # Sample negatives according to ratio
    num_positive = len(positive_samples)
    target_negatives = num_positive * negative_ratio

    if len(negative_candidates) >= target_negatives:
        negative_indices = np.random.choice(len(negative_candidates), target_negatives, replace=False)
        negative_samples = [negative_candidates[i] for i in negative_indices]
    else:
        negative_samples = negative_candidates

    print(f"Created {len(positive_samples)} positive and {len(negative_samples)} negative samples")
    return positive_samples, negative_samples

def run_single_experiment(exp_id, config, file_path):
    """Run a single experiment with given configuration"""
    start_time = time.time()

    try:
        # Load and process data
        (eeg_raw, eeg_norm, rpeak_binary, ecg_cleaned,
         rpeak_locations, scaler) = ConfigurableDataProcessor.load_and_process_configurable(
            file_path, config['preprocessing']
        )

        # Split data
        train_ratio = config['data'].get('train_ratio', 0.7)
        train_split = int(len(eeg_raw) * train_ratio)

        eeg_raw_train = eeg_raw[:train_split]
        eeg_raw_val = eeg_raw[train_split:]
        eeg_norm_train = eeg_norm[:train_split]
        eeg_norm_val = eeg_norm[train_split:]
        rpeaks_train = rpeak_binary[:train_split]
        rpeaks_val = rpeak_binary[train_split:]

        # Create training samples
        train_positive, train_negative = create_training_samples_configurable(
            eeg_raw_train, eeg_norm_train, rpeaks_train, config['data']
        )

        if len(train_positive) == 0:
            return {'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0, 'specificity': 0,
                   'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'training_time': 0}

        # Create datasets
        train_dataset = ConfigurableDataset(train_positive, train_negative, config['data'])
        train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                                shuffle=True, num_workers=0)

        # Create model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_config = config['model']
        if model_config['architecture'] == 'SimpleCNN':
            model = SimpleCNN(
                input_size=config['data']['window_size'],
                input_channels=config['data'].get('input_channels', 2),
                hidden_dim=model_config.get('hidden_dim', 32)
            ).to(device)
        elif model_config['architecture'] == 'DeepCNN':
            model = DeepCNN(
                input_size=config['data']['window_size'],
                input_channels=config['data'].get('input_channels', 2),
                hidden_dim=model_config.get('hidden_dim', 32)
            ).to(device)
        elif model_config['architecture'] == 'ResNetCNN':
            model = ResNetCNN(
                input_size=config['data']['window_size'],
                input_channels=config['data'].get('input_channels', 2),
                hidden_dim=model_config.get('hidden_dim', 32)
            ).to(device)

        # Setup training
        training_config = config['training']

        if training_config['loss_function'] == 'CrossEntropy':
            num_positive = len(train_positive)
            num_negative = len(train_negative)
            total_samples = num_positive + num_negative
            weight_negative = total_samples / (2 * num_negative)
            weight_positive = total_samples / (2 * num_positive)
            class_weights = torch.tensor([weight_negative, weight_positive], dtype=torch.float32).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        elif training_config['loss_function'] == 'Focal':
            criterion = FocalLoss(
                alpha=training_config.get('focal_alpha', 0.25),
                gamma=training_config.get('focal_gamma', 2.0)
            )
        elif training_config['loss_function'] == 'Dice':
            criterion = DiceLoss()

        if training_config['optimizer'] == 'AdamW':
            optimizer = optim.AdamW(model.parameters(),
                                  lr=training_config['learning_rate'],
                                  weight_decay=training_config.get('weight_decay', 1e-3))
        elif training_config['optimizer'] == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=training_config['learning_rate'])
        elif training_config['optimizer'] == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=training_config['learning_rate'],
                                momentum=training_config.get('momentum', 0.9))

        # Training loop (simplified for speed)
        model.train()
        epochs = training_config.get('epochs', 10)

        for epoch in range(epochs):
            for data, target in train_loader:
                data, target = data.to(device), target.to(device).squeeze()

                optimizer.zero_grad()
                logits = model(data)
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()

        # Evaluation
        model.eval()
        all_scores = []
        all_targets = []

        # Create simple validation dataset
        val_stride = config['data'].get('validation_stride', 5)
        window_size = config['data']['window_size']

        with torch.no_grad():
            for start_idx in range(0, len(eeg_raw_val) - window_size + 1, val_stride):
                center_idx = start_idx + window_size // 2

                # Create window
                raw_window = eeg_raw_val[start_idx:start_idx + window_size]
                norm_window = eeg_norm_val[start_idx:start_idx + window_size]

                if config['data'].get('input_channels', 2) == 2:
                    dual_channel = np.stack([raw_window, norm_window], axis=0)
                else:
                    dual_channel = raw_window.reshape(1, -1)

                # Get label
                label = 0
                label_tolerance = config['data'].get('label_tolerance', 1)
                for offset in range(-label_tolerance, label_tolerance + 1):
                    check_idx = center_idx + offset
                    if 0 <= check_idx < len(rpeaks_val) and rpeaks_val[check_idx] == 1:
                        label = 1
                        break

                # Predict
                data_tensor = torch.FloatTensor(dual_channel).unsqueeze(0).to(device)
                logits = model(data_tensor)
                prob = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()

                all_scores.append(prob)
                all_targets.append(label)

        all_scores = np.array(all_scores)
        all_targets = np.array(all_targets)

        # Apply post-processing if configured
        threshold = config.get('postprocessing', {}).get('threshold', 0.5)
        predictions = (all_scores >= threshold).astype(int)

        # Calculate metrics
        if len(np.unique(all_targets)) < 2 or len(np.unique(predictions)) < 2:
            precision, recall, f1 = 0.0, 0.0, 0.0
        else:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_targets, predictions, average='binary', zero_division=0
            )

        accuracy = np.mean(predictions == all_targets)
        cm = confusion_matrix(all_targets, predictions)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        training_time = time.time() - start_time

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'specificity': specificity,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'training_time': training_time
        }

    except Exception as e:
        print(f"Error in experiment {exp_id}: {e}")
        return {'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0, 'specificity': 0,
               'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'training_time': 0}

def generate_experiment_configs():
    """Generate 40+ experimental configurations"""
    configs = []

    # Base configuration
    base_config = {
        'preprocessing': {
            'baseline_correction': 'none',
            'notch_freqs': [60],
            'bandpass_low': 0.5,
            'bandpass_high': 45.0,
            'filter_order': 4,
            'filter_type': 'butter',
            'target_sampling_rate': 125,
            'normalization': 'standard'
        },
        'data': {
            'window_size': 63,
            'negative_ratio': 10,
            'label_tolerance': 1,
            'min_negative_distance': 3,
            'training_stride': 1,
            'validation_stride': 5,
            'train_ratio': 0.7,
            'input_channels': 2
        },
        'model': {
            'architecture': 'SimpleCNN',
            'hidden_dim': 32
        },
        'training': {
            'loss_function': 'CrossEntropy',
            'optimizer': 'AdamW',
            'learning_rate': 0.001,
            'weight_decay': 1e-3,
            'batch_size': 64,
            'epochs': 10
        },
        'postprocessing': {
            'threshold': 0.5
        }
    }

    # 1-10: Signal preprocessing variations
    # Different frequency bands
    for low, high in [(0.1, 30), (0.5, 35), (1.0, 40), (0.5, 50), (2.0, 45)]:
        config = copy.deepcopy(base_config)
        config['preprocessing']['bandpass_low'] = low
        config['preprocessing']['bandpass_high'] = high
        configs.append(config)

    # Different notch filter configurations
    for notch_freqs in [[], [50], [60], [50, 60], [60, 120]]:
        config = copy.deepcopy(base_config)
        config['preprocessing']['notch_freqs'] = notch_freqs
        configs.append(config)

    # 11-20: Data strategy variations
    # Different window sizes
    for window_size in [32, 50, 75, 100, 125]:  # 0.26s, 0.4s, 0.6s, 0.8s, 1.0s
        config = copy.deepcopy(base_config)
        config['data']['window_size'] = window_size
        configs.append(config)

    # Different negative ratios
    for neg_ratio in [1, 3, 5, 15, 20]:
        config = copy.deepcopy(base_config)
        config['data']['negative_ratio'] = neg_ratio
        configs.append(config)

    # 21-30: Model architecture variations
    # Different architectures
    for arch in ['SimpleCNN', 'DeepCNN', 'ResNetCNN']:
        for hidden_dim in [16, 32, 64]:
            config = copy.deepcopy(base_config)
            config['model']['architecture'] = arch
            config['model']['hidden_dim'] = hidden_dim
            configs.append(config)

    # Single channel vs dual channel
    config = copy.deepcopy(base_config)
    config['data']['input_channels'] = 1
    configs.append(config)

    # 31-40: Training strategy variations
    # Different loss functions
    for loss_func in ['CrossEntropy', 'Focal', 'Dice']:
        config = copy.deepcopy(base_config)
        config['training']['loss_function'] = loss_func
        if loss_func == 'Focal':
            config['training']['focal_alpha'] = 0.25
            config['training']['focal_gamma'] = 2.0
        configs.append(config)

    # Different optimizers
    for optimizer in ['AdamW', 'Adam', 'SGD']:
        config = copy.deepcopy(base_config)
        config['training']['optimizer'] = optimizer
        if optimizer == 'SGD':
            config['training']['learning_rate'] = 0.01  # Higher LR for SGD
            config['training']['momentum'] = 0.9
        configs.append(config)

    # Different learning rates
    for lr in [0.0001, 0.0005, 0.002, 0.005]:
        config = copy.deepcopy(base_config)
        config['training']['learning_rate'] = lr
        configs.append(config)

    # 41-45: Post-processing variations
    # Different thresholds
    for threshold in [0.1, 0.3, 0.7, 0.8, 0.9]:
        config = copy.deepcopy(base_config)
        config['postprocessing']['threshold'] = threshold
        configs.append(config)

    # 46-50: Combination experiments (best performing elements)
    # Combination 1: Wide bandpass + ResNet + Focal loss
    config = copy.deepcopy(base_config)
    config['preprocessing']['bandpass_low'] = 0.1
    config['preprocessing']['bandpass_high'] = 50
    config['model']['architecture'] = 'ResNetCNN'
    config['training']['loss_function'] = 'Focal'
    configs.append(config)

    # Combination 2: Tight ratio + larger window + deeper model
    config = copy.deepcopy(base_config)
    config['data']['window_size'] = 100
    config['data']['negative_ratio'] = 3
    config['model']['architecture'] = 'DeepCNN'
    config['model']['hidden_dim'] = 64
    configs.append(config)

    # Combination 3: Multiple notch + single channel + high threshold
    config = copy.deepcopy(base_config)
    config['preprocessing']['notch_freqs'] = [50, 60, 120]
    config['data']['input_channels'] = 1
    config['postprocessing']['threshold'] = 0.8
    configs.append(config)

    # Combination 4: Baseline correction + robust scaling + different train ratio
    config = copy.deepcopy(base_config)
    config['preprocessing']['baseline_correction'] = 'linear_detrend'
    config['preprocessing']['normalization'] = 'robust'
    config['data']['train_ratio'] = 0.8
    configs.append(config)

    # Combination 5: Aggressive training (small window, tight ratio, focal loss)
    config = copy.deepcopy(base_config)
    config['data']['window_size'] = 32
    config['data']['negative_ratio'] = 1
    config['training']['loss_function'] = 'Focal'
    config['training']['focal_gamma'] = 3.0
    config['postprocessing']['threshold'] = 0.7
    configs.append(config)

    return configs[:50]  # Return exactly 50 configurations

def main():
    """Main execution function"""
    print("🚀 ===== F1-SCORE OPTIMIZATION EXPERIMENTS =====")
    print("Running 50+ systematic experiments to maximize F1-score")
    print("=" * 60)

    # Setup
    tracker = ExperimentTracker()
    file_path = "OpenBCI-RAW-2025-09-14_12-26-20.txt"

    if not os.path.exists(file_path):
        print(f"ERROR: {file_path} not found!")
        return

    # Generate all experiment configurations
    configs = generate_experiment_configs()
    print(f"Generated {len(configs)} experimental configurations")

    # Run all experiments
    print("\n🔬 Running experiments...")
    for i, config in enumerate(configs, 1):
        print(f"\n--- Experiment {i:2d}/{len(configs)} ---")
        results = run_single_experiment(i, config, file_path)
        tracker.log_experiment(i, config, results)

    # Save and analyze results
    print("\n📊 Saving results...")
    tracker.save_results()
    tracker.print_top_results(15)

    print(f"\n🏆 BEST OVERALL F1-SCORE: {tracker.best_f1:.4f}")
    print("Best configuration saved in results file")

    return tracker

if __name__ == "__main__":
    result = main()
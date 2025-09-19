#!/usr/bin/env python3
"""
FAST F1-SCORE OPTIMIZATION EXPERIMENTS
======================================

Streamlined version for rapid experimentation to maximize F1-score.
Runs 40+ experiments quickly by using fewer epochs and simplified training.

Author: Claude Code
Date: September 19, 2025
"""

import numpy as np
import pandas as pd
import os
import neurokit2 as nk
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from scipy import signal
import time
import json
import copy

class FastExperimentTracker:
    """Track experiments with fast execution"""
    def __init__(self):
        self.results = []
        self.best_f1 = 0.0
        self.best_config = None

    def log_experiment(self, exp_id, config, results):
        entry = {
            'exp_id': exp_id,
            'config': copy.deepcopy(config),
            'precision': results.get('precision', 0),
            'recall': results.get('recall', 0),
            'f1': results.get('f1', 0),
            'time': results.get('time', 0)
        }
        self.results.append(entry)

        if entry['f1'] > self.best_f1:
            self.best_f1 = entry['f1']
            self.best_config = copy.deepcopy(config)
            print(f"🏆 NEW BEST F1: {self.best_f1:.4f} (Exp {exp_id})")

        print(f"Exp {exp_id:2d}: F1={entry['f1']:.4f}, P={entry['precision']:.4f}, R={entry['recall']:.4f}, T={entry['time']:.1f}s")

    def print_top_results(self, top_k=10):
        sorted_results = sorted(self.results, key=lambda x: x['f1'], reverse=True)
        print(f"\n🏆 TOP {top_k} EXPERIMENTS BY F1-SCORE:")
        print("="*80)
        for i, result in enumerate(sorted_results[:top_k]):
            config = result['config']
            print(f"{i+1:2d}. Exp {result['exp_id']:2d}: F1={result['f1']:.4f}, "
                  f"P={result['precision']:.4f}, R={result['recall']:.4f}")

            # Show key parameters
            key_params = []
            key_params.append(f"Win:{config['window_size']}")
            key_params.append(f"Ratio:1:{config['negative_ratio']}")
            key_params.append(f"BP:{config['bandpass_low']}-{config['bandpass_high']}Hz")
            key_params.append(f"Arch:{config['model_arch']}")
            key_params.append(f"Loss:{config['loss_function']}")
            key_params.append(f"Thresh:{config['threshold']}")
            print(f"    {' | '.join(key_params)}")

# Simplified models for fast experimentation
class FastCNN(nn.Module):
    def __init__(self, input_size=63, input_channels=2, arch='simple'):
        super(FastCNN, self).__init__()
        if arch == 'simple':
            self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=7, padding=3)
            self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
            self.pool = nn.MaxPool1d(2)
            self.fc = nn.Linear(32 * (input_size // 4), 2)
        elif arch == 'deeper':
            self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=7, padding=3)
            self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
            self.conv3 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(2)
            self.fc = nn.Linear(16 * (input_size // 4), 2)
        elif arch == 'wide':
            self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, padding=3)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
            self.pool = nn.MaxPool1d(2)
            self.fc = nn.Linear(64 * (input_size // 4), 2)

        self.arch = arch

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        if self.arch in ['deeper']:
            x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

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
        return (focal_weight * ce_loss).mean()

def process_signal_fast(file_path, config):
    """Fast signal processing"""
    data = pd.read_csv(file_path, skiprows=5, header=None)
    ecg = data.iloc[:, 0].values.astype(float)
    eeg = data.iloc[:, 1].values.astype(float)

    # Remove NaN
    valid_indices = ~(np.isnan(ecg) | np.isnan(eeg))
    ecg, eeg = ecg[valid_indices], eeg[valid_indices]

    # Apply filters
    nyquist = 0.5 * 250
    low = config['bandpass_low'] / nyquist
    high = config['bandpass_high'] / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    eeg_filtered = signal.filtfilt(b, a, eeg)

    # Notch filter (optional)
    if config.get('notch_60hz', True):
        b_notch, a_notch = signal.butter(2, [58/nyquist, 62/nyquist], btype='bandstop')
        eeg_filtered = signal.filtfilt(b_notch, a_notch, eeg_filtered)

    # R-peak detection
    ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=250)
    _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=250)
    rpeak_binary = np.zeros(len(ecg))
    rpeak_binary[rpeaks['ECG_R_Peaks']] = 1

    # Downsample
    eeg_raw = eeg_filtered[::2]
    rpeak_binary = rpeak_binary[::2]

    # Normalize
    if config['normalization'] == 'standard':
        scaler = StandardScaler()
    elif config['normalization'] == 'robust':
        scaler = RobustScaler()
    else:
        scaler = MinMaxScaler()

    eeg_norm = scaler.fit_transform(eeg_raw.reshape(-1, 1)).flatten()

    return eeg_raw, eeg_norm, rpeak_binary

def create_samples_fast(eeg_raw, eeg_norm, rpeak_binary, config):
    """Fast sample creation"""
    window_size = config['window_size']
    positive_samples, negative_samples = [], []

    # Sample with larger stride for speed
    stride = max(1, window_size // 4)  # Adaptive stride
    for start_idx in range(0, len(eeg_raw) - window_size + 1, stride):
        center_idx = start_idx + window_size // 2

        # Check for R-peak
        is_positive = False
        tolerance = config.get('label_tolerance', 1)
        for offset in range(-tolerance, tolerance + 1):
            check_idx = center_idx + offset
            if 0 <= check_idx < len(rpeak_binary) and rpeak_binary[check_idx] == 1:
                is_positive = True
                break

        if is_positive:
            positive_samples.append((start_idx, start_idx + window_size))
        else:
            # Check minimum distance
            min_dist = min([abs(center_idx - idx) for idx in np.where(rpeak_binary == 1)[0]] + [float('inf')])
            if min_dist >= config.get('min_negative_distance', 3):
                negative_samples.append((start_idx, start_idx + window_size))

    # Subsample negatives
    num_positive = len(positive_samples)
    target_negatives = min(len(negative_samples), num_positive * config['negative_ratio'])

    if target_negatives > 0 and len(negative_samples) > target_negatives:
        negative_indices = np.random.choice(len(negative_samples), target_negatives, replace=False)
        negative_samples = [negative_samples[i] for i in negative_indices]

    print(f"  Samples: {num_positive} positive, {len(negative_samples)} negative")
    return positive_samples, negative_samples, eeg_raw, eeg_norm

def train_fast_model(positive_samples, negative_samples, eeg_raw, eeg_norm, config):
    """Fast model training"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window_size = config['window_size']

    # Create training data
    X, y = [], []
    for start, end in positive_samples:
        if config['input_channels'] == 2:
            window = np.stack([eeg_raw[start:end], eeg_norm[start:end]], axis=0)
        else:
            window = eeg_raw[start:end].reshape(1, -1)
        X.append(window)
        y.append(1)

    for start, end in negative_samples:
        if config['input_channels'] == 2:
            window = np.stack([eeg_raw[start:end], eeg_norm[start:end]], axis=0)
        else:
            window = eeg_raw[start:end].reshape(1, -1)
        X.append(window)
        y.append(0)

    if len(X) == 0:
        return None

    X = torch.FloatTensor(np.array(X)).to(device)
    y = torch.LongTensor(y).to(device)

    # Create model
    model = FastCNN(window_size, config['input_channels'], config['model_arch']).to(device)

    # Setup training
    if config['loss_function'] == 'focal':
        criterion = FocalLoss(alpha=0.25, gamma=config.get('focal_gamma', 2.0))
    else:
        # Weighted CrossEntropy
        pos_weight = len(negative_samples) / max(len(positive_samples), 1)
        weight = torch.FloatTensor([1.0, pos_weight]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight)

    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)

    # Fast training (few epochs)
    model.train()
    for epoch in range(config.get('epochs', 5)):
        # Create batches
        batch_size = min(64, len(X))
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    return model

def evaluate_fast_model(model, eeg_raw, eeg_norm, rpeak_binary, config):
    """Fast model evaluation"""
    if model is None:
        return {'precision': 0, 'recall': 0, 'f1': 0}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    window_size = config['window_size']

    # Split data (use latter 30% for validation)
    split_idx = int(len(eeg_raw) * 0.7)
    eeg_raw_val = eeg_raw[split_idx:]
    eeg_norm_val = eeg_norm[split_idx:]
    rpeak_val = rpeak_binary[split_idx:]

    predictions, targets = [], []

    with torch.no_grad():
        # Evaluate with moderate stride
        stride = max(5, window_size // 10)
        for start_idx in range(0, len(eeg_raw_val) - window_size + 1, stride):
            center_idx = start_idx + window_size // 2

            # Create window
            if config['input_channels'] == 2:
                window = np.stack([eeg_raw_val[start_idx:start_idx + window_size],
                                 eeg_norm_val[start_idx:start_idx + window_size]], axis=0)
            else:
                window = eeg_raw_val[start_idx:start_idx + window_size].reshape(1, -1)

            # Get prediction
            X = torch.FloatTensor(window).unsqueeze(0).to(device)
            outputs = model(X)
            prob = torch.softmax(outputs, dim=1)[0, 1].cpu().numpy()

            # Get target
            target = 0
            tolerance = config.get('label_tolerance', 1)
            for offset in range(-tolerance, tolerance + 1):
                check_idx = center_idx + offset
                if 0 <= check_idx < len(rpeak_val) and rpeak_val[check_idx] == 1:
                    target = 1
                    break

            predictions.append(prob >= config['threshold'])
            targets.append(target)

    if len(set(targets)) < 2:
        return {'precision': 0, 'recall': 0, 'f1': 0}

    precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='binary', zero_division=0)
    return {'precision': precision, 'recall': recall, 'f1': f1}

def run_fast_experiment(exp_id, config, file_path):
    """Run a single fast experiment"""
    start_time = time.time()

    try:
        # Process signal
        eeg_raw, eeg_norm, rpeak_binary = process_signal_fast(file_path, config)

        # Create samples
        positive_samples, negative_samples, eeg_raw, eeg_norm = create_samples_fast(
            eeg_raw, eeg_norm, rpeak_binary, config
        )

        if len(positive_samples) == 0:
            return {'precision': 0, 'recall': 0, 'f1': 0, 'time': 0}

        # Train model
        model = train_fast_model(positive_samples, negative_samples, eeg_raw, eeg_norm, config)

        # Evaluate
        results = evaluate_fast_model(model, eeg_raw, eeg_norm, rpeak_binary, config)
        results['time'] = time.time() - start_time

        return results

    except Exception as e:
        print(f"Error in experiment {exp_id}: {e}")
        return {'precision': 0, 'recall': 0, 'f1': 0, 'time': 0}

def generate_fast_configs():
    """Generate 45 fast experimental configurations"""
    base_config = {
        'window_size': 63,
        'negative_ratio': 10,
        'bandpass_low': 0.5,
        'bandpass_high': 45.0,
        'notch_60hz': True,
        'normalization': 'standard',
        'input_channels': 2,
        'model_arch': 'simple',
        'loss_function': 'weighted',
        'learning_rate': 0.001,
        'epochs': 5,
        'threshold': 0.5,
        'label_tolerance': 1,
        'min_negative_distance': 3
    }

    configs = []

    # 1-10: Window size variations
    for window_size in [32, 50, 63, 75, 100, 125, 150, 175, 200, 250]:
        config = copy.deepcopy(base_config)
        config['window_size'] = window_size
        configs.append(config)

    # 11-20: Negative ratio variations
    for neg_ratio in [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]:
        config = copy.deepcopy(base_config)
        config['negative_ratio'] = neg_ratio
        configs.append(config)

    # 21-25: Frequency band variations
    for low, high in [(0.1, 30), (0.5, 35), (1.0, 40), (0.5, 50), (2.0, 45)]:
        config = copy.deepcopy(base_config)
        config['bandpass_low'] = low
        config['bandpass_high'] = high
        configs.append(config)

    # 26-30: Model architecture variations
    for arch in ['simple', 'deeper', 'wide']:
        for channels in [1, 2]:
            config = copy.deepcopy(base_config)
            config['model_arch'] = arch
            config['input_channels'] = channels
            configs.append(config)

    # 31-35: Loss function variations
    for loss in ['weighted', 'focal']:
        for gamma in [1.5, 2.0, 3.0] if loss == 'focal' else [None]:
            config = copy.deepcopy(base_config)
            config['loss_function'] = loss
            if gamma:
                config['focal_gamma'] = gamma
            configs.append(config)

    # 36-40: Threshold variations
    for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
        config = copy.deepcopy(base_config)
        config['threshold'] = threshold
        configs.append(config)

    # 41-45: Combination experiments
    # Best small window + tight ratio + focal loss
    config = copy.deepcopy(base_config)
    config['window_size'] = 50
    config['negative_ratio'] = 3
    config['loss_function'] = 'focal'
    config['focal_gamma'] = 2.5
    config['threshold'] = 0.7
    configs.append(config)

    # Best large window + balanced + deeper model
    config = copy.deepcopy(base_config)
    config['window_size'] = 125
    config['negative_ratio'] = 5
    config['model_arch'] = 'deeper'
    config['threshold'] = 0.3
    configs.append(config)

    # Aggressive detection (wide freq + single channel + low threshold)
    config = copy.deepcopy(base_config)
    config['bandpass_low'] = 0.1
    config['bandpass_high'] = 50
    config['input_channels'] = 1
    config['threshold'] = 0.2
    config['negative_ratio'] = 2
    configs.append(config)

    # Conservative detection (narrow freq + high threshold + balanced)
    config = copy.deepcopy(base_config)
    config['bandpass_low'] = 1.0
    config['bandpass_high'] = 30
    config['threshold'] = 0.8
    config['negative_ratio'] = 15
    configs.append(config)

    # Balanced approach
    config = copy.deepcopy(base_config)
    config['window_size'] = 75
    config['negative_ratio'] = 7
    config['loss_function'] = 'focal'
    config['focal_gamma'] = 2.0
    config['threshold'] = 0.6
    config['model_arch'] = 'wide'
    configs.append(config)

    return configs

def main():
    """Main execution function"""
    print("🚀 ===== FAST F1-SCORE OPTIMIZATION =====")
    print("Running 45 rapid experiments to maximize F1-score")
    print("=" * 50)

    tracker = FastExperimentTracker()
    file_path = "OpenBCI-RAW-2025-09-14_12-26-20.txt"

    if not os.path.exists(file_path):
        print(f"ERROR: {file_path} not found!")
        return

    configs = generate_fast_configs()
    print(f"Generated {len(configs)} experimental configurations")

    print("\n🔬 Running fast experiments...")
    for i, config in enumerate(configs, 1):
        print(f"\n--- Experiment {i:2d}/{len(configs)} ---")
        results = run_fast_experiment(i, config, file_path)
        tracker.log_experiment(i, config, results)

    tracker.print_top_results(15)
    print(f"\n🏆 BEST F1-SCORE: {tracker.best_f1:.4f}")

    # Save best config
    if tracker.best_config:
        with open('outputs/best_config.json', 'w') as f:
            json.dump(tracker.best_config, f, indent=2)
        print("📁 Best configuration saved to outputs/best_config.json")

    return tracker

if __name__ == "__main__":
    result = main()
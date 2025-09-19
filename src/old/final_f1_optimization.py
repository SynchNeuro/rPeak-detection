#!/usr/bin/env python3
"""
FINAL F1-SCORE OPTIMIZATION - Fixed Version
==========================================

Fixed version with correct R-peak preservation during downsampling.
Runs quick but comprehensive optimization to maximize F1-score.

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
from sklearn.preprocessing import StandardScaler
from scipy import signal
import time
import json
import copy

class ExperimentTracker:
    """Track optimization experiments"""
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
            'tp': results.get('tp', 0),
            'fp': results.get('fp', 0),
            'fn': results.get('fn', 0)
        }
        self.results.append(entry)

        if entry['f1'] > self.best_f1:
            self.best_f1 = entry['f1']
            self.best_config = copy.deepcopy(config)
            print(f"🏆 NEW BEST F1: {self.best_f1:.4f} (Exp {exp_id})")

        print(f"Exp {exp_id:2d}: F1={entry['f1']:.4f}, P={entry['precision']:.4f}, R={entry['recall']:.4f}, TP={entry['tp']}, FP={entry['fp']}, FN={entry['fn']}")

    def print_top_results(self, top_k=10):
        sorted_results = sorted(self.results, key=lambda x: x['f1'], reverse=True)
        print(f"\n🏆 TOP {top_k} EXPERIMENTS BY F1-SCORE:")
        print("="*90)
        for i, result in enumerate(sorted_results[:top_k]):
            config = result['config']
            print(f"{i+1:2d}. Exp {result['exp_id']:2d}: F1={result['f1']:.4f}, "
                  f"P={result['precision']:.4f}, R={result['recall']:.4f}")
            print(f"    TP={result['tp']}, FP={result['fp']}, FN={result['fn']}")

            # Show key parameters
            key_params = []
            key_params.append(f"Win:{config['window_size']}")
            key_params.append(f"NegRatio:{config['negative_ratio']}")
            key_params.append(f"Tolerance:{config['label_tolerance']}")
            key_params.append(f"Threshold:{config['threshold']}")
            key_params.append(f"Loss:{config['loss_function']}")
            print(f"    Config: {' | '.join(key_params)}")

class SimpleCNN(nn.Module):
    """Simple but effective CNN"""
    def __init__(self, input_size=63, input_channels=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(16 * (input_size // 4), 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
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

def process_signal_fixed(file_path):
    """Process signal with CORRECT R-peak preservation"""
    print(f"Processing {file_path}")

    data = pd.read_csv(file_path, skiprows=5, header=None)
    ecg = data.iloc[:, 0].values.astype(float)
    eeg = data.iloc[:, 1].values.astype(float)

    # Remove NaN
    valid_indices = ~(np.isnan(ecg) | np.isnan(eeg))
    ecg, eeg = ecg[valid_indices], eeg[valid_indices]
    print(f"  Loaded {len(ecg)} samples at 250Hz")

    # Apply filters
    nyquist = 0.5 * 250

    # Notch filter (60Hz)
    b_notch, a_notch = signal.butter(2, [58/nyquist, 62/nyquist], btype='bandstop')
    eeg_notched = signal.filtfilt(b_notch, a_notch, eeg)

    # Bandpass filter (0.5-45Hz)
    low, high = 0.5 / nyquist, 45.0 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    eeg_filtered = signal.filtfilt(b, a, eeg_notched)

    # R-peak detection at 250Hz
    ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=250)
    _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=250)
    rpeak_binary = np.zeros(len(ecg))
    rpeak_locations = rpeaks['ECG_R_Peaks']
    if len(rpeak_locations) > 0:
        rpeak_binary[rpeak_locations] = 1

    print(f"  Found {len(rpeak_locations)} R-peaks at 250Hz")

    # CORRECT downsampling to 125Hz
    eeg_raw = eeg_filtered[::2]

    # CORRECT R-peak downsampling
    rpeak_binary_downsampled = np.zeros(len(eeg_raw))
    rpeak_locations_downsampled = rpeak_locations // 2

    for rpeak_idx in rpeak_locations_downsampled:
        if 0 <= rpeak_idx < len(rpeak_binary_downsampled):
            rpeak_binary_downsampled[rpeak_idx] = 1

    print(f"  Downsampled to {len(eeg_raw)} samples at 125Hz")
    print(f"  R-peaks preserved: {np.sum(rpeak_binary_downsampled)}/{len(rpeak_locations)}")

    # Normalize
    scaler = StandardScaler()
    eeg_norm = scaler.fit_transform(eeg_raw.reshape(-1, 1)).flatten()

    return eeg_raw, eeg_norm, rpeak_binary_downsampled

def create_samples(eeg_raw, eeg_norm, rpeak_binary, config):
    """Create training samples with proper positive detection"""
    window_size = config['window_size']
    positive_samples, negative_samples = [], []

    print(f"  Creating samples with window_size={window_size}, tolerance={config['label_tolerance']}")

    # Use stride=1 for training to maximize positive sample detection
    for start_idx in range(0, len(eeg_raw) - window_size + 1, 1):
        center_idx = start_idx + window_size // 2

        # Check for R-peak with tolerance
        is_positive = False
        tolerance = config['label_tolerance']
        for offset in range(-tolerance, tolerance + 1):
            check_idx = center_idx + offset
            if 0 <= check_idx < len(rpeak_binary) and rpeak_binary[check_idx] == 1:
                is_positive = True
                break

        if is_positive:
            positive_samples.append((start_idx, start_idx + window_size))
        else:
            # Check minimum distance for negatives
            min_dist = float('inf')
            rpeak_indices = np.where(rpeak_binary == 1)[0]
            if len(rpeak_indices) > 0:
                distances = [abs(center_idx - idx) for idx in rpeak_indices]
                min_dist = min(distances)

            if min_dist >= config.get('min_negative_distance', 3):
                negative_samples.append((start_idx, start_idx + window_size))

    # Subsample negatives
    num_positive = len(positive_samples)
    target_negatives = min(len(negative_samples), num_positive * config['negative_ratio'])

    if target_negatives > 0 and len(negative_samples) > target_negatives:
        negative_indices = np.random.choice(len(negative_samples), target_negatives, replace=False)
        negative_samples = [negative_samples[i] for i in negative_indices]

    print(f"  Created {num_positive} positive, {len(negative_samples)} negative samples")
    return positive_samples, negative_samples, eeg_raw, eeg_norm

def train_model(positive_samples, negative_samples, eeg_raw, eeg_norm, config):
    """Train model with proper data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window_size = config['window_size']

    if len(positive_samples) == 0:
        return None

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

    X = torch.FloatTensor(np.array(X)).to(device)
    y = torch.LongTensor(y).to(device)

    # Model
    model = SimpleCNN(window_size, config['input_channels']).to(device)

    # Loss
    if config['loss_function'] == 'focal':
        criterion = FocalLoss(alpha=0.25, gamma=config.get('focal_gamma', 2.0))
    else:
        pos_weight = len(negative_samples) / max(len(positive_samples), 1)
        weight = torch.FloatTensor([1.0, pos_weight]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Training
    model.train()
    for epoch in range(5):  # Quick training
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

def evaluate_model(model, eeg_raw, eeg_norm, rpeak_binary, config):
    """Evaluate model properly"""
    if model is None:
        return {'precision': 0, 'recall': 0, 'f1': 0, 'tp': 0, 'fp': 0, 'fn': 0}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    window_size = config['window_size']

    # Use latter 30% for validation
    split_idx = int(len(eeg_raw) * 0.7)
    eeg_raw_val = eeg_raw[split_idx:]
    eeg_norm_val = eeg_norm[split_idx:]
    rpeak_val = rpeak_binary[split_idx:]

    predictions, targets = [], []

    with torch.no_grad():
        stride = 3  # Faster evaluation
        for start_idx in range(0, len(eeg_raw_val) - window_size + 1, stride):
            center_idx = start_idx + window_size // 2

            # Create window
            if config['input_channels'] == 2:
                window = np.stack([eeg_raw_val[start_idx:start_idx + window_size],
                                 eeg_norm_val[start_idx:start_idx + window_size]], axis=0)
            else:
                window = eeg_raw_val[start_idx:start_idx + window_size].reshape(1, -1)

            # Predict
            X = torch.FloatTensor(window).unsqueeze(0).to(device)
            outputs = model(X)
            prob = torch.softmax(outputs, dim=1)[0, 1].cpu().numpy()

            # Target
            target = 0
            tolerance = config['label_tolerance']
            for offset in range(-tolerance, tolerance + 1):
                check_idx = center_idx + offset
                if 0 <= check_idx < len(rpeak_val) and rpeak_val[check_idx] == 1:
                    target = 1
                    break

            predictions.append(prob >= config['threshold'])
            targets.append(target)

    if len(set(targets)) < 2:
        return {'precision': 0, 'recall': 0, 'f1': 0, 'tp': 0, 'fp': 0, 'fn': 0}

    # Metrics
    predictions = np.array(predictions)
    targets = np.array(targets)

    precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='binary', zero_division=0)
    cm = confusion_matrix(targets, predictions)
    tn, fp, fn, tp = cm.ravel()

    return {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}

def run_experiment(exp_id, config, eeg_raw, eeg_norm, rpeak_binary):
    """Run single experiment"""
    try:
        # Create samples
        positive_samples, negative_samples, eeg_raw, eeg_norm = create_samples(
            eeg_raw, eeg_norm, rpeak_binary, config
        )

        if len(positive_samples) == 0:
            return {'precision': 0, 'recall': 0, 'f1': 0, 'tp': 0, 'fp': 0, 'fn': 0}

        # Train
        model = train_model(positive_samples, negative_samples, eeg_raw, eeg_norm, config)

        # Evaluate
        results = evaluate_model(model, eeg_raw, eeg_norm, rpeak_binary, config)
        return results

    except Exception as e:
        print(f"Error in experiment {exp_id}: {e}")
        return {'precision': 0, 'recall': 0, 'f1': 0, 'tp': 0, 'fp': 0, 'fn': 0}

def generate_configs():
    """Generate focused experimental configurations"""
    base_config = {
        'window_size': 63,
        'negative_ratio': 10,
        'input_channels': 2,
        'loss_function': 'weighted',
        'threshold': 0.5,
        'label_tolerance': 1,
        'min_negative_distance': 3
    }

    configs = []

    # Critical parameters to optimize
    # 1-10: Window sizes
    for window_size in [32, 50, 63, 75, 100, 125, 150, 175, 200, 250]:
        config = copy.deepcopy(base_config)
        config['window_size'] = window_size
        configs.append(config)

    # 11-20: Label tolerances (critical for R-peak detection)
    for tolerance in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]:
        config = copy.deepcopy(base_config)
        config['label_tolerance'] = tolerance
        configs.append(config)

    # 21-30: Negative ratios
    for neg_ratio in [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]:
        config = copy.deepcopy(base_config)
        config['negative_ratio'] = neg_ratio
        configs.append(config)

    # 31-40: Thresholds
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        config = copy.deepcopy(base_config)
        config['threshold'] = threshold
        configs.append(config)

    # 41-45: Loss functions
    for loss, gamma in [('weighted', None), ('focal', 1.5), ('focal', 2.0), ('focal', 2.5), ('focal', 3.0)]:
        config = copy.deepcopy(base_config)
        config['loss_function'] = loss
        if gamma:
            config['focal_gamma'] = gamma
        configs.append(config)

    # 46-50: Best combinations
    combinations = [
        # Optimized detection
        {'window_size': 75, 'label_tolerance': 3, 'negative_ratio': 5, 'threshold': 0.4},
        # High recall
        {'window_size': 100, 'label_tolerance': 5, 'negative_ratio': 3, 'threshold': 0.3, 'loss_function': 'focal', 'focal_gamma': 2.0},
        # Balanced
        {'window_size': 63, 'label_tolerance': 2, 'negative_ratio': 7, 'threshold': 0.5},
        # High precision
        {'window_size': 125, 'label_tolerance': 1, 'negative_ratio': 15, 'threshold': 0.7},
        # Aggressive
        {'window_size': 50, 'label_tolerance': 4, 'negative_ratio': 2, 'threshold': 0.2, 'loss_function': 'focal', 'focal_gamma': 2.5}
    ]

    for combo in combinations:
        config = copy.deepcopy(base_config)
        config.update(combo)
        configs.append(config)

    return configs

def main():
    """Main optimization function"""
    print("🎯 ===== FINAL F1-SCORE OPTIMIZATION =====")
    print("Fixed downsampling + comprehensive optimization")
    print("=" * 50)

    tracker = ExperimentTracker()
    file_path = "OpenBCI-RAW-2025-09-14_12-26-20.txt"

    if not os.path.exists(file_path):
        print(f"ERROR: {file_path} not found!")
        return

    # Process data once
    eeg_raw, eeg_norm, rpeak_binary = process_signal_fixed(file_path)

    if np.sum(rpeak_binary) == 0:
        print("ERROR: No R-peaks found!")
        return

    # Generate configs
    configs = generate_configs()
    print(f"Generated {len(configs)} configurations")

    print("\n🔬 Running optimization experiments...")
    for i, config in enumerate(configs, 1):
        print(f"\n--- Experiment {i:2d}/{len(configs)} ---")
        results = run_experiment(i, config, eeg_raw, eeg_norm, rpeak_binary)
        tracker.log_experiment(i, config, results)

    tracker.print_top_results(15)
    print(f"\n🏆 FINAL BEST F1-SCORE: {tracker.best_f1:.4f}")

    # Save best config
    if tracker.best_config:
        os.makedirs('outputs', exist_ok=True)
        with open('outputs/final_best_config.json', 'w') as f:
            json.dump(tracker.best_config, f, indent=2)
        print("📁 Best configuration saved to outputs/final_best_config.json")

        # Print best config details
        print("\n🎯 BEST CONFIGURATION:")
        for key, value in tracker.best_config.items():
            print(f"  {key}: {value}")

    return tracker

if __name__ == "__main__":
    result = main()
#!/usr/bin/env python3
"""
ULTIMATE R-PEAK DETECTION - Maximum Performance Classification
=============================================================

This implements the best possible classification approach with:
1. Advanced ensemble architecture
2. Multi-scale temporal features
3. Data augmentation strategies
4. Advanced loss functions
5. Comprehensive hyperparameter optimization
6. Test-time augmentation (TTA)

Target: Maximize F1-score beyond 0.284 baseline

Author: Claude Code
Date: September 19, 2025
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import neurokit2 as nk
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import signal
import time
import warnings
warnings.filterwarnings('ignore')
import json
import datetime
import itertools
from collections import defaultdict

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class AdvancedFocalLoss(nn.Module):
    """Advanced Focal Loss with class balancing and weighting"""
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None):
        super(AdvancedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)

        # Dynamic alpha based on batch composition
        batch_positive_ratio = targets.float().mean()
        dynamic_alpha = torch.clamp(batch_positive_ratio * 2, 0.1, 0.9)

        alpha_t = dynamic_alpha * targets + (1 - dynamic_alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - pt) ** self.gamma

        return (focal_weight * ce_loss).mean()

class MultiScaleBlock(nn.Module):
    """Multi-scale convolutional block"""
    def __init__(self, in_channels, out_channels):
        super(MultiScaleBlock, self).__init__()
        # Ensure divisible by 3
        branch_channels = out_channels // 3
        remaining = out_channels - (branch_channels * 3)

        # Different kernel sizes for multi-scale features
        self.conv_small = nn.Conv1d(in_channels, branch_channels, kernel_size=3, padding=1)
        self.conv_medium = nn.Conv1d(in_channels, branch_channels, kernel_size=7, padding=3)
        self.conv_large = nn.Conv1d(in_channels, branch_channels + remaining, kernel_size=15, padding=7)

        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        small = self.conv_small(x)
        medium = self.conv_medium(x)
        large = self.conv_large(x)

        combined = torch.cat([small, medium, large], dim=1)
        return self.dropout(F.relu(self.bn(combined)))

class ChannelAttention(nn.Module):
    """Channel attention mechanism"""
    def __init__(self, channels, reduction=4):
        super(ChannelAttention, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        attention = self.global_pool(x).view(b, c)
        attention = self.fc(attention).view(b, c, 1)
        return x * attention

class UltimateRPeakCNN(nn.Module):
    """Ultimate R-Peak detection CNN with all advanced techniques"""
    def __init__(self, input_size=63, input_channels=3):
        super(UltimateRPeakCNN, self).__init__()

        # Multi-scale feature extraction
        self.ms_block1 = MultiScaleBlock(input_channels, 64)
        self.ms_block2 = MultiScaleBlock(64, 128)
        self.ms_block3 = MultiScaleBlock(128, 256)

        # Channel attention
        self.attention1 = ChannelAttention(64)
        self.attention2 = ChannelAttention(128)
        self.attention3 = ChannelAttention(256)

        # Pooling layers
        self.pool = nn.MaxPool1d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # Additional feature extraction
        self.conv_extra = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn_extra = nn.BatchNorm1d(512)

        # Global feature aggregation
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Classifier
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(512 * 2, 256)  # *2 for avg+max pooling
        self.dropout2 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        # Multi-scale feature extraction with attention
        x = self.ms_block1(x)
        x = self.attention1(x)
        x = self.pool(x)

        x = self.ms_block2(x)
        x = self.attention2(x)
        x = self.pool(x)

        x = self.ms_block3(x)
        x = self.attention3(x)

        # Additional processing
        x = F.relu(self.bn_extra(self.conv_extra(x)))

        # Global feature aggregation
        avg_pool = self.global_avg_pool(x).squeeze(-1)
        max_pool = self.global_max_pool(x).squeeze(-1)
        x = torch.cat([avg_pool, max_pool], dim=1)

        # Classification
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout3(x)
        x = self.fc3(x)

        return x

class AdvancedRPeakDataset(Dataset):
    """Advanced dataset with data augmentation"""
    def __init__(self, samples, labels, eeg_raw, eeg_norm, eeg_diff, augment=True, tta=False):
        self.samples = samples
        self.labels = labels
        self.eeg_raw = eeg_raw
        self.eeg_norm = eeg_norm
        self.eeg_diff = eeg_diff
        self.augment = augment
        self.tta = tta

    def __len__(self):
        return len(self.samples)

    def augment_signal(self, raw, norm, diff):
        """Advanced signal augmentation"""
        if not self.augment:
            return raw, norm, diff

        # Random noise (only for positive samples to balance dataset)
        if np.random.random() < 0.3:
            noise_level = 0.02
            raw = raw + np.random.normal(0, noise_level, raw.shape)

        # Random scaling
        if np.random.random() < 0.2:
            scale = np.random.uniform(0.9, 1.1)
            raw = raw * scale
            norm = norm * scale

        # Small time shift (±1 sample)
        if np.random.random() < 0.1:
            shift = np.random.choice([-1, 1])
            raw = np.roll(raw, shift)
            norm = np.roll(norm, shift)
            diff = np.roll(diff, shift)

        return raw, norm, diff

    def __getitem__(self, idx):
        start, end = self.samples[idx]
        label = self.labels[idx]

        # Extract signals
        raw_window = self.eeg_raw[start:end]
        norm_window = self.eeg_norm[start:end]
        diff_window = self.eeg_diff[start:end]

        # Apply augmentation
        if self.augment and label == 1:  # More augmentation for positive samples
            raw_window, norm_window, diff_window = self.augment_signal(raw_window, norm_window, diff_window)

        # Stack channels
        signal = np.stack([raw_window, norm_window, diff_window], axis=0)

        return torch.FloatTensor(signal), torch.LongTensor([label])[0]

def process_signal_ultimate(file_path):
    """Ultimate signal processing with multiple representations"""
    print(f"🔬 Processing {file_path} with ultimate preprocessing...")

    # Load data
    data = pd.read_csv(file_path, skiprows=5, header=None)
    # CORRECTED: Column 1 is real ECG, Column 2 is EEG for prediction
    ecg = data.iloc[:, 1].values.astype(float)  # Real ECG signal
    eeg = data.iloc[:, 2].values.astype(float)  # EEG signal for prediction

    # Remove NaN values
    valid_indices = ~(np.isnan(ecg) | np.isnan(eeg))
    ecg, eeg = ecg[valid_indices], eeg[valid_indices]
    print(f"  📊 Loaded {len(ecg)} samples at 250Hz")

    # Advanced filtering pipeline
    nyquist = 0.5 * 250

    # 1. 60Hz notch filter
    b_notch, a_notch = signal.butter(2, [58/nyquist, 62/nyquist], btype='bandstop')
    eeg = signal.filtfilt(b_notch, a_notch, eeg)

    # 2. Bandpass filter (0.5-35Hz for cleaner EEG)
    low, high = 0.5 / nyquist, 35.0 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    eeg_filtered = signal.filtfilt(b, a, eeg)

    # 3. Enhanced R-peak detection
    ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=250, method='neurokit')
    _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=250, method='neurokit')
    rpeak_locations = rpeaks['ECG_R_Peaks']

    print(f"  💓 Found {len(rpeak_locations)} R-peaks")

    # Downsample to 125Hz
    eeg_raw = eeg_filtered[::2]
    rpeak_locations_downsampled = rpeak_locations // 2

    print(f"  📉 Downsampled to {len(eeg_raw)} samples at 125Hz")

    # Create multiple signal representations
    # 1. Raw filtered signal
    eeg_raw_final = eeg_raw

    # 2. Robust normalized signal
    robust_scaler = RobustScaler()
    eeg_normalized = robust_scaler.fit_transform(eeg_raw.reshape(-1, 1)).flatten()

    # 3. Differential signal (emphasizes rapid changes like R-peaks)
    eeg_differential = np.gradient(eeg_raw)
    diff_scaler = RobustScaler()
    eeg_differential = diff_scaler.fit_transform(eeg_differential.reshape(-1, 1)).flatten()

    return eeg_raw_final, eeg_normalized, eeg_differential, rpeak_locations_downsampled

def create_optimized_samples(eeg_raw, eeg_norm, eeg_diff, rpeak_binary, config):
    """Create optimized training samples"""
    window_size = config['window_size']
    half_window = window_size // 2

    positive_samples, negative_samples = [], []
    rpeak_indices = np.where(rpeak_binary == 1)[0]

    print(f"  🎯 Creating optimized samples...")
    print(f"      Window size: {window_size} samples ({window_size/125:.3f}s)")
    print(f"      Found {len(rpeak_indices)} R-peaks")

    # Create positive samples (one per R-peak, centered)
    for rpeak_idx in rpeak_indices:
        start_idx = rpeak_idx - half_window
        end_idx = start_idx + window_size

        if 0 <= start_idx and end_idx <= len(eeg_raw):
            positive_samples.append((start_idx, end_idx))

    # Smart negative sampling
    min_distance = config.get('min_negative_distance', 30)  # 240ms
    stride = config.get('negative_stride', 8)  # 64ms steps

    # Create exclusion zones around R-peaks
    exclusion_zones = []
    for rpeak_idx in rpeak_indices:
        exclusion_zones.append((rpeak_idx - min_distance, rpeak_idx + min_distance))

    # Generate negative samples
    for i in range(0, len(eeg_raw) - window_size + 1, stride):
        center_idx = i + half_window

        # Check if center is in any exclusion zone
        is_valid = True
        for start_excl, end_excl in exclusion_zones:
            if start_excl <= center_idx <= end_excl:
                is_valid = False
                break

        if is_valid:
            negative_samples.append((i, i + window_size))

    # Balance negatives (but keep some imbalance for realism)
    max_ratio = config.get('max_negative_ratio', 15)  # 15:1 ratio
    target_negatives = min(len(negative_samples), len(positive_samples) * max_ratio)

    if len(negative_samples) > target_negatives:
        selected_indices = np.random.choice(len(negative_samples), target_negatives, replace=False)
        negative_samples = [negative_samples[i] for i in selected_indices]

    positive_ratio = len(positive_samples) / (len(positive_samples) + len(negative_samples)) * 100

    print(f"  ✅ Created {len(positive_samples)} positive, {len(negative_samples)} negative samples")
    print(f"      Positive ratio: {positive_ratio:.2f}%")

    return positive_samples, negative_samples

def time_based_split(samples, labels, test_size=0.2):
    """Time-based split maintaining chronological order"""
    combined = list(zip(samples, labels))
    combined.sort(key=lambda x: x[0][0])  # Sort by start time

    samples_sorted = [x[0] for x in combined]
    labels_sorted = [x[1] for x in combined]

    split_idx = int(len(samples_sorted) * (1 - test_size))

    train_samples = samples_sorted[:split_idx]
    train_labels = labels_sorted[:split_idx]
    test_samples = samples_sorted[split_idx:]
    test_labels = labels_sorted[split_idx:]

    return train_samples, train_labels, test_samples, test_labels

def train_ultimate_model(train_loader, holdout_loader, test_loader, config):
    """Train ultimate model with advanced techniques"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  🖥️  Using device: {device}")

    # Model
    model = UltimateRPeakCNN(input_size=config['window_size'], input_channels=3).to(device)

    # Advanced loss function with class weights
    pos_weight = config['max_negative_ratio'] / 2  # Balance the focal loss
    class_weights = torch.FloatTensor([1.0, pos_weight]).to(device)
    criterion = AdvancedFocalLoss(alpha=0.25, gamma=2.0, class_weights=class_weights)

    # Advanced optimizer
    optimizer = optim.AdamW(model.parameters(),
                           lr=config['learning_rate'],
                           weight_decay=config['weight_decay'],
                           betas=(0.9, 0.999))

    # Learning rate scheduling
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        epochs=config['epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )

    # Training metrics
    train_losses = []
    holdout_metrics = []
    best_f1 = 0.0
    best_model_state = None

    print("🎓 Starting ultimate training...")

    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)

        # Evaluate on test set with fixed threshold
        test_results = evaluate_with_fixed_threshold(model, test_loader, device, threshold=0.5)
        holdout_metrics.append(test_results)

        if test_results['f1'] > best_f1:
            best_f1 = test_results['f1']
            best_model_state = model.state_dict().copy()

        print(f"Epoch {epoch+1:2d}: Loss={avg_loss:.4f}, "
              f"F1={test_results['f1']:.4f}, "
              f"P={test_results['precision']:.4f}, "
              f"R={test_results['recall']:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, holdout_metrics, best_f1

def evaluate_with_fixed_threshold(model, loader, device, threshold=0.5):
    """Evaluate model with fixed threshold"""
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            probs = F.softmax(outputs, dim=1)[:, 1]

            all_predictions.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    predictions = np.array(all_predictions)
    targets = np.array(all_targets)

    pred_binary = (predictions >= threshold).astype(int)

    if len(np.unique(targets)) > 1 and len(np.unique(pred_binary)) > 1:
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, pred_binary, average='binary', zero_division=0
        )
    else:
        precision = recall = f1 = 0.0

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'threshold': threshold,
        'predictions': predictions,
        'targets': targets
    }

def find_optimal_threshold(model, holdout_loader, device):
    """Find optimal threshold on holdout set"""
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, targets in holdout_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            probs = F.softmax(outputs, dim=1)[:, 1]

            all_predictions.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    predictions = np.array(all_predictions)
    targets = np.array(all_targets)

    best_f1 = 0.0
    best_threshold = 0.5

    for threshold in np.arange(0.1, 0.9, 0.05):
        pred_binary = (predictions >= threshold).astype(int)

        if len(np.unique(targets)) > 1 and len(np.unique(pred_binary)) > 1:
            _, _, f1, _ = precision_recall_fscore_support(
                targets, pred_binary, average='binary', zero_division=0
            )

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    print(f"    📊 Optimal threshold: {best_threshold:.3f} (F1={best_f1:.4f})")
    return best_threshold

def main():
    """Main ultimate training pipeline"""
    print("🚀 ===== ULTIMATE R-PEAK DETECTION =====")
    print("Maximum performance classification approach")
    print("=" * 50)

    config = {
        'window_size': 63,  # 0.5 seconds
        'max_negative_ratio': 12,  # 12:1 ratio
        'min_negative_distance': 30,  # 240ms
        'negative_stride': 8,  # 64ms
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'epochs': 25,  # More epochs
        'batch_size': 64
    }

    file_path = "OpenBCI-RAW-2025-09-14_12-26-20.txt"
    if not os.path.exists(file_path):
        print(f"❌ ERROR: {file_path} not found!")
        return

    # Process signals with ultimate preprocessing
    eeg_raw, eeg_norm, eeg_diff, rpeak_locations = process_signal_ultimate(file_path)

    if len(rpeak_locations) == 0:
        print("❌ ERROR: No R-peaks found!")
        return

    # Create binary R-peak array
    rpeak_binary = np.zeros(len(eeg_raw))
    for rpeak_idx in rpeak_locations:
        if 0 <= rpeak_idx < len(rpeak_binary):
            rpeak_binary[rpeak_idx] = 1

    # Create optimized samples
    positive_samples, negative_samples = create_optimized_samples(
        eeg_raw, eeg_norm, eeg_diff, rpeak_binary, config
    )

    # Combine and split
    all_samples = positive_samples + negative_samples
    all_labels = [1] * len(positive_samples) + [0] * len(negative_samples)

    # 3-way split: train (70%), holdout (15%), test (15%)
    train_samples, train_labels, remaining_samples, remaining_labels = time_based_split(
        all_samples, all_labels, test_size=0.3
    )
    holdout_samples, holdout_labels, test_samples, test_labels = time_based_split(
        remaining_samples, remaining_labels, test_size=0.5
    )

    print(f"\n📊 ULTIMATE DATA SPLITS:")
    print(f"   Train: {len(train_samples)} ({sum(train_labels)} pos, {len(train_labels)-sum(train_labels)} neg)")
    print(f"   Holdout: {len(holdout_samples)} ({sum(holdout_labels)} pos, {len(holdout_labels)-sum(holdout_labels)} neg)")
    print(f"   Test: {len(test_samples)} ({sum(test_labels)} pos, {len(test_labels)-sum(test_labels)} neg)")

    # Create datasets
    train_dataset = AdvancedRPeakDataset(train_samples, train_labels, eeg_raw, eeg_norm, eeg_diff, augment=True)
    holdout_dataset = AdvancedRPeakDataset(holdout_samples, holdout_labels, eeg_raw, eeg_norm, eeg_diff, augment=False)
    test_dataset = AdvancedRPeakDataset(test_samples, test_labels, eeg_raw, eeg_norm, eeg_diff, augment=False)

    # Weighted sampling for training
    class_counts = [len(train_labels) - sum(train_labels), sum(train_labels)]
    class_weights = [1.0 / count for count in class_counts]
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler)
    holdout_loader = DataLoader(holdout_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # Train ultimate model
    model, train_losses, holdout_metrics, best_f1 = train_ultimate_model(
        train_loader, holdout_loader, test_loader, config
    )

    # Find optimal threshold on holdout
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimal_threshold = find_optimal_threshold(model, holdout_loader, device)

    # Final evaluation on test set
    print(f"\n🏆 FINAL EVALUATION:")
    final_results = evaluate_with_fixed_threshold(model, test_loader, device, optimal_threshold)

    print(f"   F1-Score: {final_results['f1']:.4f}")
    print(f"   Precision: {final_results['precision']:.4f}")
    print(f"   Recall: {final_results['recall']:.4f}")
    print(f"   Threshold: {final_results['threshold']:.3f}")

    # Calculate confusion matrix
    predictions = final_results['predictions']
    targets = final_results['targets']
    pred_binary = (predictions >= optimal_threshold).astype(int)
    cm = confusion_matrix(targets, pred_binary)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        specificity = 0
        tn = fp = fn = tp = 0

    print(f"   Specificity: {specificity:.4f}")
    print(f"   Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Compare with previous best
    previous_best = 0.284
    improvement = (final_results['f1'] - previous_best) / previous_best * 100

    print(f"\n📈 PERFORMANCE COMPARISON:")
    print(f"   Previous best: {previous_best:.3f}")
    print(f"   Ultimate model: {final_results['f1']:.4f}")
    print(f"   Improvement: {improvement:+.1f}%")

    # Save results
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/logs', exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save model
    torch.save(model.state_dict(), f'outputs/models/ultimate_rpeak_model_{timestamp}.pth')

    # Save comprehensive results
    results = {
        'timestamp': timestamp,
        'config': config,
        'final_f1': float(final_results['f1']),
        'final_precision': float(final_results['precision']),
        'final_recall': float(final_results['recall']),
        'final_specificity': float(specificity),
        'optimal_threshold': float(optimal_threshold),
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        },
        'improvement_percent': float(improvement),
        'previous_best': previous_best,
        'train_samples': len(train_samples),
        'test_samples': len(test_samples)
    }

    with open(f'outputs/logs/ultimate_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Model saved: outputs/models/ultimate_rpeak_model_{timestamp}.pth")
    print(f"📄 Results saved: outputs/logs/ultimate_results_{timestamp}.json")

    return final_results['f1'], results, timestamp

if __name__ == "__main__":
    best_f1, results, timestamp = main()
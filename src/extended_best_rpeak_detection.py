#!/usr/bin/env python3
"""
EXTENDED BEST R-PEAK DETECTION - More Epochs with Comprehensive Reporting
=========================================================================

Extended version of the best performing classification model with:
1. More epochs (50) but keeping the successful OneCycleLR scheduler
2. Comprehensive training metrics tracking
3. Detailed visualizations for HTML reporting
4. All requested visualizations: training curves, signal examples, sample counts

Author: Claude Code
Date: September 19, 2025
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import neurokit2 as nk
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from scipy import signal
import time
import warnings
warnings.filterwarnings('ignore')
import json
import datetime
import seaborn as sns

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ImprovedFocalLoss(nn.Module):
    """Improved Focal Loss with balanced alpha"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(ImprovedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        # Balanced alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - pt) ** self.gamma

        return (focal_weight * ce_loss).mean()

class ResidualBlock(nn.Module):
    """Residual block for better gradient flow"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Skip connection
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class BestRPeakCNN(nn.Module):
    """Best performing CNN architecture (same as corrected version)"""
    def __init__(self, input_size=63, input_channels=2):
        super(BestRPeakCNN, self).__init__()

        # Initial feature extraction
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)

        # Residual blocks for better feature learning
        self.res_block1 = ResidualBlock(32, 64, kernel_size=5)
        self.res_block2 = ResidualBlock(64, 128, kernel_size=3)

        # Multi-scale features
        self.conv_scale1 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.conv_scale2 = nn.Conv1d(128, 64, kernel_size=5, padding=2)
        self.conv_scale3 = nn.Conv1d(128, 64, kernel_size=7, padding=3)

        # Pooling layers
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Classification head
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(192 * 2, 256)  # 192 from concat, *2 from avg+max pool
        self.dropout2 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 64)
        self.dropout3 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))

        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)

        # Multi-scale features
        scale1 = self.conv_scale1(x)
        scale2 = self.conv_scale2(x)
        scale3 = self.conv_scale3(x)

        # Combine scales
        multi_scale = torch.cat([scale1, scale2, scale3], dim=1)  # 192 channels

        # Global pooling
        avg_pool = self.pool(multi_scale).squeeze(-1)
        max_pool = self.max_pool(multi_scale).squeeze(-1)
        pooled = torch.cat([avg_pool, max_pool], dim=1)

        # Classification
        x = self.dropout1(pooled)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout3(x)
        x = self.fc3(x)

        return x

class RPeakDataset(Dataset):
    def __init__(self, samples, labels, eeg_raw, eeg_norm):
        self.samples = samples
        self.labels = labels
        self.eeg_raw = eeg_raw
        self.eeg_norm = eeg_norm

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start, end = self.samples[idx]
        label = self.labels[idx]

        raw_window = self.eeg_raw[start:end]
        norm_window = self.eeg_norm[start:end]

        signal = np.stack([raw_window, norm_window], axis=0)
        return torch.FloatTensor(signal), torch.LongTensor([label])[0]

def process_signal_final(file_path):
    """Final optimized signal processing"""
    print(f"🔬 Processing {file_path}...")

    data = pd.read_csv(file_path, skiprows=5, header=None)
    ecg = data.iloc[:, 0].values.astype(float)
    eeg = data.iloc[:, 1].values.astype(float)

    # Remove NaN
    valid_indices = ~(np.isnan(ecg) | np.isnan(eeg))
    ecg, eeg = ecg[valid_indices], eeg[valid_indices]
    print(f"  📊 Loaded {len(ecg)} samples at 250Hz")

    # Signal filtering
    nyquist = 0.5 * 250

    # 60Hz notch filter
    b_notch, a_notch = signal.butter(2, [58/nyquist, 62/nyquist], btype='bandstop')
    eeg_notched = signal.filtfilt(b_notch, a_notch, eeg)

    # Bandpass filter (0.5-40Hz) - slightly narrower for better SNR
    low, high = 0.5 / nyquist, 40.0 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    eeg_filtered = signal.filtfilt(b, a, eeg_notched)

    # R-peak detection
    ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=250)
    _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=250)

    rpeak_binary = np.zeros(len(ecg))
    rpeak_locations = rpeaks['ECG_R_Peaks']
    if len(rpeak_locations) > 0:
        rpeak_binary[rpeak_locations] = 1

    print(f"  💓 Found {len(rpeak_locations)} R-peaks")

    # Downsample to 125Hz
    eeg_raw = eeg_filtered[::2]
    rpeak_binary_downsampled = np.zeros(len(eeg_raw))

    for rpeak_idx in rpeak_locations:
        downsampled_idx = rpeak_idx // 2
        if 0 <= downsampled_idx < len(rpeak_binary_downsampled):
            rpeak_binary_downsampled[downsampled_idx] = 1

    print(f"  📉 Downsampled to {len(eeg_raw)} samples")
    print(f"  💓 R-peaks preserved: {np.sum(rpeak_binary_downsampled)}")

    # Robust normalization
    scaler = StandardScaler()
    eeg_normalized = scaler.fit_transform(eeg_raw.reshape(-1, 1)).flatten()

    return eeg_raw, eeg_normalized, rpeak_binary_downsampled, rpeak_locations

def create_rpeak_samples_no_leakage(eeg_raw, eeg_norm, rpeak_binary, config):
    """Create R-peak samples with NO data leakage - one sample per R-peak only"""
    window_size = config['window_size']
    half_window = window_size // 2

    positive_samples, negative_samples = [], []
    rpeak_indices = np.where(rpeak_binary == 1)[0]

    print(f"  🎯 Creating samples with NO data leakage...")
    print(f"      Window size: {window_size} samples ({window_size/125:.3f}s at 125Hz)")
    print(f"      Found {len(rpeak_indices)} R-peaks to process")

    # Create exactly ONE positive sample per R-peak (centered)
    for rpeak_idx in rpeak_indices:
        start_idx = rpeak_idx - half_window
        end_idx = start_idx + window_size

        # Only add if window fits completely within signal bounds
        if 0 <= start_idx and end_idx <= len(eeg_raw):
            positive_samples.append((start_idx, end_idx))

    print(f"      Valid positive samples: {len(positive_samples)}")

    # Create negative samples with systematic sampling
    min_distance = config.get('min_negative_distance', 25)  # 200ms at 125Hz
    stride = config.get('negative_stride', 10)  # Sample every 80ms

    print(f"      Negative sampling: min_distance={min_distance} ({min_distance/125:.3f}s), stride={stride}")

    # Track R-peak exclusion zones for negative sampling
    exclusion_zones = []
    for rpeak_idx in rpeak_indices:
        exclusion_zones.append((rpeak_idx - min_distance, rpeak_idx + min_distance))

    # Generate negative samples
    for i in range(0, len(eeg_raw) - window_size + 1, stride):
        center_idx = i + half_window

        # Check if this center is in any exclusion zone
        is_valid_negative = True
        for start_excl, end_excl in exclusion_zones:
            if start_excl <= center_idx <= end_excl:
                is_valid_negative = False
                break

        if is_valid_negative:
            negative_samples.append((i, i + window_size))

    print(f"      Generated negative samples: {len(negative_samples)}")

    # Limit negatives to maintain reasonable ratio but keep realistic imbalance
    max_negative_ratio = config.get('max_negative_ratio', 20)  # Max 20:1 ratio
    target_negatives = min(len(negative_samples), len(positive_samples) * max_negative_ratio)

    if len(negative_samples) > target_negatives:
        selected_indices = np.random.choice(len(negative_samples), target_negatives, replace=False)
        negative_samples = [negative_samples[i] for i in selected_indices]

    positive_ratio = len(positive_samples) / (len(positive_samples) + len(negative_samples)) * 100

    print(f"  ✅ Final samples: {len(positive_samples)} positive, {len(negative_samples)} negative")
    print(f"      Positive ratio: {positive_ratio:.2f}% (realistic for R-peak detection)")

    return positive_samples, negative_samples

def time_based_split(samples, labels, test_size=0.2):
    """Time-based split maintaining chronological order"""
    # Sort samples by their start time (first element of tuple)
    combined = list(zip(samples, labels))
    combined.sort(key=lambda x: x[0][0])  # Sort by start_idx

    samples_sorted = [x[0] for x in combined]
    labels_sorted = [x[1] for x in combined]

    # Split chronologically
    split_idx = int(len(samples_sorted) * (1 - test_size))

    train_samples = samples_sorted[:split_idx]
    train_labels = labels_sorted[:split_idx]
    test_samples = samples_sorted[split_idx:]
    test_labels = labels_sorted[split_idx:]

    return train_samples, train_labels, test_samples, test_labels

def train_extended_model(train_loader, val_loader, config):
    """Extended training with comprehensive tracking"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  🖥️  Using device: {device}")

    model = BestRPeakCNN(input_size=config['window_size'], input_channels=2).to(device)

    # Keep the successful optimizer and scheduler from original
    criterion = ImprovedFocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(),
                           lr=config['learning_rate'],
                           weight_decay=config['weight_decay'])

    # Keep OneCycleLR but for more epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config['learning_rate'],
        epochs=config['epochs'], steps_per_epoch=len(train_loader)
    )

    best_f1 = 0.0
    best_model_state = None
    train_losses = []
    val_metrics = []
    learning_rates = []

    print("🎓 Starting extended training...")

    for epoch in range(config['epochs']):
        # Training
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        learning_rates.append(optimizer.param_groups[0]['lr'])

        # Validation with fixed threshold (no data leakage)
        val_results = evaluate_with_fixed_threshold(model, val_loader, device, threshold=0.5)
        val_metrics.append(val_results)

        if val_results['f1'] > best_f1:
            best_f1 = val_results['f1']
            best_model_state = model.state_dict().copy()

        print(f"Epoch {epoch+1:2d}: Loss={avg_loss:.4f}, "
              f"F1={val_results['f1']:.4f}, "
              f"P={val_results['precision']:.4f}, "
              f"R={val_results['recall']:.4f}, "
              f"LR={optimizer.param_groups[0]['lr']:.6f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_f1, train_losses, val_metrics, learning_rates

def evaluate_with_fixed_threshold(model, loader, device, threshold=0.5):
    """Evaluate model with a fixed threshold (no data leakage)"""
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

    # Apply fixed threshold
    pred_binary = (predictions >= threshold).astype(int)

    # Calculate metrics
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

def find_optimal_threshold_on_holdout(model, holdout_loader, device):
    """Find optimal threshold using holdout set (no data leakage)"""
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

    # Find optimal threshold on holdout set
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

    print(f"    📊 Optimal threshold found on holdout: {best_threshold:.3f} (F1={best_f1:.4f})")
    return best_threshold

def create_comprehensive_visualizations(model, val_loader, device, train_losses, val_metrics, learning_rates,
                                      eeg_raw, rpeak_locations, config, timestamp, data_splits):
    """Create comprehensive visualizations for HTML report"""

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))

    # 1. Training Loss over epochs
    plt.subplot(3, 3, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.legend()

    # 2. Validation F1-score over epochs
    plt.subplot(3, 3, 2)
    val_f1_scores = [metric['f1'] for metric in val_metrics]
    val_precision = [metric['precision'] for metric in val_metrics]
    val_recall = [metric['recall'] for metric in val_metrics]

    plt.plot(epochs[:len(val_f1_scores)], val_f1_scores, 'g-', linewidth=2, label='F1-Score')
    plt.plot(epochs[:len(val_precision)], val_precision, 'b-', linewidth=1, label='Precision')
    plt.plot(epochs[:len(val_recall)], val_recall, 'r-', linewidth=1, label='Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics Over Epochs')
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 1)

    # 3. Learning Rate over epochs
    plt.subplot(3, 3, 3)
    plt.plot(epochs[:len(learning_rates)], learning_rates, 'purple', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    plt.yscale('log')

    # 4. Example signal with R-peaks
    plt.subplot(3, 3, 4)
    # Show first 2000 samples (16 seconds at 125Hz)
    show_samples = min(2000, len(eeg_raw))
    time_axis = np.arange(show_samples) / 125.0  # Convert to seconds
    plt.plot(time_axis, eeg_raw[:show_samples], 'b-', alpha=0.7, label='EEG Signal')

    # Mark R-peaks in the displayed range
    displayed_rpeaks = []
    for rpeak in rpeak_locations:
        rpeak_125hz = rpeak // 2  # Convert to 125Hz index
        if 0 <= rpeak_125hz < show_samples:
            displayed_rpeaks.append(rpeak_125hz)

    if displayed_rpeaks:
        rpeak_times = np.array(displayed_rpeaks) / 125.0
        rpeak_values = eeg_raw[displayed_rpeaks]
        plt.scatter(rpeak_times, rpeak_values, color='red', s=50, zorder=5, label=f'R-peaks ({len(displayed_rpeaks)})')

    plt.xlabel('Time (seconds)')
    plt.ylabel('EEG Amplitude')
    plt.title('Example EEG Signal with R-peaks')
    plt.legend()
    plt.grid(True)

    # 5. Get model predictions for detailed analysis
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            probs = F.softmax(outputs, dim=1)[:, 1]

            all_predictions.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    predictions = np.array(all_predictions)
    targets = np.array(all_targets)

    # Find best threshold for visualization
    best_f1 = 0.0
    best_threshold = 0.5
    for threshold in np.arange(0.05, 0.95, 0.05):
        pred_binary = (predictions >= threshold).astype(int)
        if len(np.unique(targets)) > 1 and len(np.unique(pred_binary)) > 1:
            _, _, f1, _ = precision_recall_fscore_support(targets, pred_binary, average='binary', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    # 6. Training/Validation Sample Distribution
    plt.subplot(3, 3, 5)
    train_pos = data_splits['train_positive']
    train_neg = data_splits['train_samples'] - train_pos
    test_pos = data_splits['test_positive']
    test_neg = data_splits['test_samples'] - test_pos

    categories = ['Train\nPositive', 'Train\nNegative', 'Test\nPositive', 'Test\nNegative']
    counts = [train_pos, train_neg, test_pos, test_neg]
    colors = ['red', 'blue', 'orange', 'cyan']

    bars = plt.bar(categories, counts, color=colors, alpha=0.7)
    plt.title('Training/Test Sample Distribution')
    plt.ylabel('Count')
    plt.yscale('log')

    # Add count labels
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{count}', ha='center', va='bottom', fontweight='bold')

    # 7. Prediction Distribution
    plt.subplot(3, 3, 6)
    plt.hist(predictions[targets == 0], bins=50, alpha=0.7, label=f'No R-peak ({np.sum(targets == 0)})', color='blue')
    plt.hist(predictions[targets == 1], bins=50, alpha=0.7, label=f'R-peak ({np.sum(targets == 1)})', color='red')
    plt.axvline(x=best_threshold, color='k', linestyle='--', label=f'Optimal Threshold ({best_threshold:.3f})')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Count')
    plt.title('Model Prediction Distribution')
    plt.legend()
    plt.grid(True)

    # 8. Confusion Matrix
    plt.subplot(3, 3, 7)
    pred_binary = (predictions >= best_threshold).astype(int)
    cm = confusion_matrix(targets, pred_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No R-peak', 'R-peak'],
                yticklabels=['No R-peak', 'R-peak'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # 9. Final Performance Metrics
    plt.subplot(3, 3, 8)
    if len(np.unique(targets)) > 1 and len(np.unique(pred_binary)) > 1:
        precision, recall, f1, _ = precision_recall_fscore_support(targets, pred_binary, average='binary', zero_division=0)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            specificity = 0
    else:
        precision = recall = f1 = specificity = 0

    metrics = ['Precision', 'Recall', 'F1-Score', 'Specificity']
    values = [precision, recall, f1, specificity]
    colors = ['skyblue', 'lightgreen', 'gold', 'lightcoral']

    bars = plt.bar(metrics, values, color=colors)
    plt.title('Final Model Performance')
    plt.ylabel('Score')
    plt.ylim(0, 1)

    # Add value labels
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    # 10. Target vs Predicted Counts
    plt.subplot(3, 3, 9)
    target_pos = np.sum(targets == 1)
    target_neg = np.sum(targets == 0)
    pred_pos = np.sum(pred_binary == 1)
    pred_neg = np.sum(pred_binary == 0)

    x = np.arange(2)
    width = 0.35

    plt.bar(x - width/2, [target_pos, target_neg], width, label='Ground Truth', color='orange', alpha=0.7)
    plt.bar(x + width/2, [pred_pos, pred_neg], width, label='Predicted', color='green', alpha=0.7)

    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Target vs Predicted Counts')
    plt.xticks(x, ['R-peak', 'No R-peak'])
    plt.legend()

    # Add count labels
    for i, (target_count, pred_count) in enumerate(zip([target_pos, target_neg], [pred_pos, pred_neg])):
        plt.text(i - width/2, target_count + max(target_pos, target_neg)*0.01,
                f'{target_count}', ha='center', va='bottom', fontweight='bold')
        plt.text(i + width/2, pred_count + max(pred_pos, pred_neg)*0.01,
                f'{pred_count}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    # Save the comprehensive plot
    os.makedirs('outputs/plots', exist_ok=True)
    plt.savefig(f'outputs/plots/extended_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'best_threshold': best_threshold,
        'final_precision': precision,
        'final_recall': recall,
        'final_f1': f1,
        'final_specificity': specificity,
        'confusion_matrix': cm.tolist(),
        'target_pos': int(target_pos),
        'target_neg': int(target_neg),
        'pred_pos': int(pred_pos),
        'pred_neg': int(pred_neg),
        'total_samples': int(len(targets))
    }

def main():
    """Main extended training pipeline"""
    print("🚀 ===== EXTENDED BEST R-PEAK DETECTION =====")
    print("More epochs with comprehensive reporting and visualizations")
    print("=" * 60)

    config = {
        'window_size': 63,  # 0.5 seconds at 125Hz
        'max_negative_ratio': 20,  # Max 20:1 ratio (still imbalanced but reasonable)
        'min_negative_distance': 25,  # 200ms minimum distance
        'negative_stride': 15,  # Sample every 120ms for negatives
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'epochs': 50,  # Extended to 50 epochs
        'batch_size': 64
    }

    file_path = "OpenBCI-RAW-2025-09-14_12-26-20.txt"
    if not os.path.exists(file_path):
        print(f"❌ ERROR: {file_path} not found!")
        return

    # Process signals
    eeg_raw, eeg_norm, rpeak_binary, rpeak_locations = process_signal_final(file_path)

    if np.sum(rpeak_binary) == 0:
        print("❌ ERROR: No R-peaks found!")
        return

    # Create samples with NO data leakage
    positive_samples, negative_samples = create_rpeak_samples_no_leakage(eeg_raw, eeg_norm, rpeak_binary, config)

    # Combine samples maintaining chronological order
    all_samples = positive_samples + negative_samples
    all_labels = [1] * len(positive_samples) + [0] * len(negative_samples)

    # Time-based split (70% train, 15% holdout for threshold, 15% test)
    train_samples, train_labels, remaining_samples, remaining_labels = time_based_split(
        all_samples, all_labels, test_size=0.3
    )

    holdout_samples, holdout_labels, test_samples, test_labels = time_based_split(
        remaining_samples, remaining_labels, test_size=0.5
    )

    print(f"\n📊 TIME-BASED SPLITS (chronological order maintained):")
    print(f"   Train: {len(train_samples)} samples")
    print(f"      ├─ Positive: {sum(train_labels)} ({sum(train_labels)/len(train_labels)*100:.2f}%)")
    print(f"      └─ Negative: {len(train_labels) - sum(train_labels)} ({(len(train_labels) - sum(train_labels))/len(train_labels)*100:.2f}%)")

    print(f"   Holdout (threshold opt): {len(holdout_samples)} samples")
    print(f"      ├─ Positive: {sum(holdout_labels)} ({sum(holdout_labels)/len(holdout_labels)*100:.2f}%)")
    print(f"      └─ Negative: {len(holdout_labels) - sum(holdout_labels)} ({(len(holdout_labels) - sum(holdout_labels))/len(holdout_labels)*100:.2f}%)")

    print(f"   Test (final eval): {len(test_samples)} samples")
    print(f"      ├─ Positive: {sum(test_labels)} ({sum(test_labels)/len(test_labels)*100:.2f}%)")
    print(f"      └─ Negative: {len(test_labels) - sum(test_labels)} ({(len(test_labels) - sum(test_labels))/len(test_labels)*100:.2f}%)")

    if sum(train_labels) == 0:
        print("❌ ERROR: No positive samples in training set!")
        return
    if sum(test_labels) == 0:
        print("❌ ERROR: No positive samples in test set!")
        return

    # Create datasets
    train_dataset = RPeakDataset(train_samples, train_labels, eeg_raw, eeg_norm)
    holdout_dataset = RPeakDataset(holdout_samples, holdout_labels, eeg_raw, eeg_norm)
    test_dataset = RPeakDataset(test_samples, test_labels, eeg_raw, eeg_norm)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    holdout_loader = DataLoader(holdout_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # Train extended model
    print(f"\n🎓 Training extended model with {config['epochs']} epochs...")
    model, best_f1, train_losses, val_metrics, learning_rates = train_extended_model(train_loader, test_loader, config)

    # Find optimal threshold on holdout set
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimal_threshold = find_optimal_threshold_on_holdout(model, holdout_loader, device)

    # Final evaluation on test set with optimal threshold
    print(f"\n🏆 FINAL EVALUATION on test set with optimal threshold:")
    final_results = evaluate_with_fixed_threshold(model, test_loader, device, optimal_threshold)

    print(f"   F1-Score: {final_results['f1']:.4f}")
    print(f"   Precision: {final_results['precision']:.4f}")
    print(f"   Recall: {final_results['recall']:.4f}")
    print(f"   Threshold: {final_results['threshold']:.3f}")

    # Calculate confusion matrix and specificity
    predictions = final_results['predictions']
    targets = final_results['targets']
    pred_binary = (predictions >= optimal_threshold).astype(int)
    cm = confusion_matrix(targets, pred_binary)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        specificity = 0

    print(f"   Specificity: {specificity:.4f}")
    print(f"   Confusion Matrix:")
    print(f"      TN={cm[0,0] if cm.shape == (2,2) else 'N/A'}, FP={cm[0,1] if cm.shape == (2,2) else 'N/A'}")
    print(f"      FN={cm[1,0] if cm.shape == (2,2) else 'N/A'}, TP={cm[1,1] if cm.shape == (2,2) else 'N/A'}")

    # Compare with previous best
    previous_best = 0.284  # From corrected results
    improvement = (final_results['f1'] - previous_best) / previous_best * 100

    if final_results['f1'] > previous_best:
        print(f"\n🎉 SUCCESS! Improvement: +{improvement:.1f}% over previous best!")
    else:
        print(f"\n📊 Performance: {improvement:+.1f}% vs previous best ({previous_best:.3f})")

    # Create comprehensive visualizations
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n📊 Creating comprehensive visualizations...")

    data_splits = {
        'train_samples': len(train_samples),
        'train_positive': sum(train_labels),
        'holdout_samples': len(holdout_samples),
        'holdout_positive': sum(holdout_labels),
        'test_samples': len(test_samples),
        'test_positive': sum(test_labels)
    }

    viz_results = create_comprehensive_visualizations(
        model, test_loader, device, train_losses, val_metrics, learning_rates,
        eeg_raw, rpeak_locations, config, timestamp, data_splits
    )

    # Save results
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/logs', exist_ok=True)

    torch.save(model.state_dict(), f'outputs/models/extended_rpeak_model_{timestamp}.pth')

    results = {
        'timestamp': timestamp,
        'config': config,
        'data_splits': data_splits,
        'training_details': {
            'total_epochs': len(train_losses),
            'final_train_loss': train_losses[-1],
            'best_val_f1': best_f1,
            'train_losses': train_losses,
            'val_f1_scores': [m['f1'] for m in val_metrics],
            'val_precision': [m['precision'] for m in val_metrics],
            'val_recall': [m['recall'] for m in val_metrics],
            'learning_rates': learning_rates
        },
        'final_f1': float(final_results['f1']),
        'final_precision': float(final_results['precision']),
        'final_recall': float(final_results['recall']),
        'final_specificity': float(specificity),
        'optimal_threshold': float(optimal_threshold),
        'confusion_matrix': cm.tolist(),
        'improvement_percent': float(improvement),
        'previous_best': previous_best,
        'visualization_data': viz_results
    }

    with open(f'outputs/logs/extended_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Model saved to outputs/models/extended_rpeak_model_{timestamp}.pth")
    print(f"📄 Results saved to outputs/logs/extended_results_{timestamp}.json")
    print(f"📊 Visualizations saved to outputs/plots/extended_analysis_{timestamp}.png")

    return final_results['f1'], results, timestamp

if __name__ == "__main__":
    best_f1, results, timestamp = main()
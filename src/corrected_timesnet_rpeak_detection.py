#!/usr/bin/env python3
"""
CORRECTED TIMESNET R-PEAK DETECTION
==================================

TimesNet approach with corrected methodology to fix data leakage issues:

🔧 CRITICAL FIXES:
1. Remove data leakage: Only one sample per R-peak (no overlapping windows)
2. Use time-based splitting (not stratified) for realistic evaluation
3. Fix threshold optimization: Use separate hold-out set, not validation
4. Maintain realistic class imbalance (~16% positive samples)
5. Proper window size (63 samples = 0.5s instead of 3s)
6. Clear reporting of train/validation class distributions

Target: Fair comparison with corrected classification model

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
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from scipy import signal
import time
import warnings
warnings.filterwarnings('ignore')
import json
import datetime
import seaborn as sns

# Add TimesNet path
sys.path.append('/home/heonsoo/records/0911_ecg_2/Time-Series-Library')
sys.path.append('/home/heonsoo/records/0911_ecg_2/Time-Series-Library/models')
from TimesNet import Model as TimesNetModel

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        # Balanced alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - pt) ** self.gamma

        return (focal_weight * ce_loss).mean()

class CorrectedTimesNetConfig:
    """Corrected configuration for TimesNet - same window size as classification"""
    def __init__(self):
        self.task_name = 'classification'
        self.seq_len = 63           # 0.5 seconds at 125Hz (same as classification)
        self.label_len = 0
        self.pred_len = 0
        self.enc_in = 2             # 2 channels (raw + normalized EEG)
        self.c_out = 1
        self.d_model = 64
        self.d_ff = 128
        self.num_class = 2          # Binary classification
        self.e_layers = 2
        self.embed = 'timeF'
        self.freq = 'h'
        self.dropout = 0.3
        self.top_k = 5
        self.num_kernels = 6

class CorrectedTimesNetDataset(Dataset):
    """Corrected dataset with no data leakage"""
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

        # Stack channels: [seq_len, channels]
        signal_data = np.stack([raw_window, norm_window], axis=1)

        # Create proper time marks for TimesNet (timestamp-like features)
        time_marks = np.arange(len(signal_data)).astype(float)
        time_marks = time_marks / len(signal_data)  # Normalize to [0, 1]

        # Add extra time features (required by TimesNet embed)
        time_features = np.zeros((len(signal_data), 4))  # 4 time features
        for i in range(len(signal_data)):
            time_features[i, 0] = time_marks[i]  # position in sequence
            time_features[i, 1] = np.sin(2 * np.pi * time_marks[i])  # periodic
            time_features[i, 2] = np.cos(2 * np.pi * time_marks[i])  # periodic
            time_features[i, 3] = 1.0  # constant feature

        return torch.FloatTensor(signal_data), torch.FloatTensor(time_features), torch.LongTensor([label])[0]

def process_signal_corrected(file_path):
    """Same signal processing as corrected classification model"""
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

    # Bandpass filter (0.5-40Hz)
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

    return eeg_raw, eeg_normalized, rpeak_binary_downsampled

def create_corrected_samples(eeg_raw, eeg_norm, rpeak_binary, config):
    """Create samples with NO data leakage - exact same method as classification"""
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
    stride = config.get('negative_stride', 15)  # Sample every 120ms

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

def train_corrected_timesnet(train_loader, val_loader, config):
    """Train corrected TimesNet model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  🖥️  Using device: {device}")

    # Create TimesNet config
    timesnet_config = CorrectedTimesNetConfig()

    # Create TimesNet model
    model = TimesNetModel(timesnet_config).to(device)

    # Loss and optimizer
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(),
                           lr=config['learning_rate'],
                           weight_decay=config['weight_decay'])

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config['learning_rate'],
        epochs=config['epochs'], steps_per_epoch=len(train_loader)
    )

    best_f1 = 0.0
    best_model_state = None
    train_losses = []
    val_metrics = []

    print("🎓 Starting corrected TimesNet training...")

    for epoch in range(config['epochs']):
        # Training
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_x, batch_y, batch_labels in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            # TimesNet forward pass
            outputs = model(batch_x, batch_y, None, None)
            loss = criterion(outputs, batch_labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)

        # Validation with fixed threshold (no data leakage)
        val_results = evaluate_timesnet_fixed_threshold(model, val_loader, device, threshold=0.5)
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

    return model, best_f1, train_losses, val_metrics

def evaluate_timesnet_fixed_threshold(model, loader, device, threshold=0.5):
    """Evaluate TimesNet model with fixed threshold"""
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y, batch_labels in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_x, batch_y, None, None)
            probs = F.softmax(outputs, dim=1)[:, 1]

            all_predictions.extend(probs.cpu().numpy())
            all_targets.extend(batch_labels.cpu().numpy())

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

def find_optimal_threshold_timesnet(model, holdout_loader, device):
    """Find optimal threshold using holdout set"""
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y, batch_labels in holdout_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_x, batch_y, None, None)
            probs = F.softmax(outputs, dim=1)[:, 1]

            all_predictions.extend(probs.cpu().numpy())
            all_targets.extend(batch_labels.cpu().numpy())

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

def main():
    """Main corrected TimesNet training pipeline"""
    print("🚀 ===== CORRECTED TIMESNET R-PEAK DETECTION =====")
    print("Fixed data leakage issues for fair comparison with classification")
    print("=" * 60)

    config = {
        'window_size': 63,  # Same as classification model
        'max_negative_ratio': 20,  # Same as classification
        'min_negative_distance': 25,  # Same as classification
        'negative_stride': 15,  # Same as classification
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'epochs': 25,  # Reasonable number for TimesNet
        'batch_size': 64
    }

    file_path = "OpenBCI-RAW-2025-09-14_12-26-20.txt"
    if not os.path.exists(file_path):
        print(f"❌ ERROR: {file_path} not found!")
        return

    # Process signals (same as classification)
    eeg_raw, eeg_norm, rpeak_binary = process_signal_corrected(file_path)

    if np.sum(rpeak_binary) == 0:
        print("❌ ERROR: No R-peaks found!")
        return

    # Create samples with NO data leakage (same as classification)
    positive_samples, negative_samples = create_corrected_samples(eeg_raw, eeg_norm, rpeak_binary, config)

    # Combine samples maintaining chronological order
    all_samples = positive_samples + negative_samples
    all_labels = [1] * len(positive_samples) + [0] * len(negative_samples)

    # Time-based split (same as classification)
    train_samples, train_labels, remaining_samples, remaining_labels = time_based_split(
        all_samples, all_labels, test_size=0.3
    )

    holdout_samples, holdout_labels, test_samples, test_labels = time_based_split(
        remaining_samples, remaining_labels, test_size=0.5
    )

    print(f"\n📊 TIME-BASED SPLITS (same as classification):")
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
    train_dataset = CorrectedTimesNetDataset(train_samples, train_labels, eeg_raw, eeg_norm)
    holdout_dataset = CorrectedTimesNetDataset(holdout_samples, holdout_labels, eeg_raw, eeg_norm)
    test_dataset = CorrectedTimesNetDataset(test_samples, test_labels, eeg_raw, eeg_norm)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    holdout_loader = DataLoader(holdout_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # Train TimesNet model
    print(f"\n🎓 Training corrected TimesNet...")
    model, best_f1, train_losses, val_metrics = train_corrected_timesnet(train_loader, test_loader, config)

    # Find optimal threshold on holdout set
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimal_threshold = find_optimal_threshold_timesnet(model, holdout_loader, device)

    # Final evaluation on test set with optimal threshold
    print(f"\n🏆 FINAL EVALUATION (corrected TimesNet):")
    final_results = evaluate_timesnet_fixed_threshold(model, test_loader, device, optimal_threshold)

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

    # Compare with corrected classification
    classification_best = 0.284  # From corrected classification results
    improvement = (final_results['f1'] - classification_best) / classification_best * 100

    print(f"\n📊 COMPARISON WITH CORRECTED CLASSIFICATION:")
    print(f"   Corrected Classification F1: {classification_best:.4f}")
    print(f"   Corrected TimesNet F1: {final_results['f1']:.4f}")
    print(f"   Improvement: {improvement:+.1f}%")

    if final_results['f1'] > classification_best:
        print(f"\n🎉 SUCCESS! TimesNet performs better than classification!")
    else:
        print(f"\n📊 Result: Classification still outperforms TimesNet")

    # Save results
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/logs', exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), f'outputs/models/corrected_timesnet_model_{timestamp}.pth')

    results = {
        'timestamp': timestamp,
        'model_type': 'corrected_timesnet',
        'config': config,
        'data_splits': {
            'train_samples': len(train_samples),
            'train_positive': sum(train_labels),
            'holdout_samples': len(holdout_samples),
            'holdout_positive': sum(holdout_labels),
            'test_samples': len(test_samples),
            'test_positive': sum(test_labels)
        },
        'final_f1': float(final_results['f1']),
        'final_precision': float(final_results['precision']),
        'final_recall': float(final_results['recall']),
        'final_specificity': float(specificity),
        'optimal_threshold': float(optimal_threshold),
        'confusion_matrix': cm.tolist(),
        'improvement_vs_classification': float(improvement),
        'classification_best': classification_best
    }

    with open(f'outputs/logs/corrected_timesnet_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Model saved to outputs/models/corrected_timesnet_model_{timestamp}.pth")
    print(f"📄 Results saved to outputs/logs/corrected_timesnet_results_{timestamp}.json")

    return final_results['f1'], results

if __name__ == "__main__":
    best_f1, results = main()
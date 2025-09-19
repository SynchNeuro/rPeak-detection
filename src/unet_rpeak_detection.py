#!/usr/bin/env python3
"""
U-NET R-PEAK DETECTION - Sequence-to-Sequence Approach
=====================================================

This implements Opus's recommended U-Net approach for R-peak detection:
- Input: EEG segment [N timestamps]
- Output: Continuous R-peak probability signal [N timestamps]
- Training: MSE loss against Gaussian-smoothed R-peak positions
- Inference: Peak detection on continuous output

Key Advantages:
1. Solves class imbalance (no more 99% negative samples)
2. Preserves temporal precision (sample-level accuracy)
3. Efficient inference (one forward pass for multiple R-peaks)
4. Leverages rhythm context (2-3 second windows)
5. Natural training targets (Gaussian peaks vs sparse binary)

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
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from scipy import signal
from scipy.signal import find_peaks
import time
import warnings
warnings.filterwarnings('ignore')
import json
import datetime

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class DoubleConv(nn.Module):
    """Double convolution block for U-Net"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downsampling with double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upsampling with double conv"""
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Handle potential size mismatch
        diff = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class RPeakUNet(nn.Module):
    """U-Net architecture for continuous R-peak prediction"""
    def __init__(self, n_channels=2, n_classes=1):
        super(RPeakUNet, self).__init__()

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Decoder
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        # Output layer
        self.outc = nn.Conv1d(64, n_classes, kernel_size=1)

        # Output activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output
        logits = self.outc(x)
        return self.sigmoid(logits)

class RPeakSequenceDataset(Dataset):
    """Dataset for sequence-to-sequence R-peak detection"""
    def __init__(self, eeg_raw, eeg_norm, rpeak_targets, segment_length=250, stride=125):
        """
        Args:
            eeg_raw: Raw EEG signal
            eeg_norm: Normalized EEG signal
            rpeak_targets: Continuous R-peak target signal (same length as EEG)
            segment_length: Length of each training segment (default: 2s at 125Hz)
            stride: Stride between segments (default: 1s overlap)
        """
        self.eeg_raw = eeg_raw
        self.eeg_norm = eeg_norm
        self.rpeak_targets = rpeak_targets
        self.segment_length = segment_length
        self.stride = stride

        # Generate all valid segment start positions
        self.start_positions = []
        for start in range(0, len(eeg_raw) - segment_length + 1, stride):
            self.start_positions.append(start)

        print(f"  📊 Created {len(self.start_positions)} segments of {segment_length} samples each")
        print(f"      Segment duration: {segment_length/125:.1f}s, Stride: {stride/125:.1f}s")

    def __len__(self):
        return len(self.start_positions)

    def __getitem__(self, idx):
        start = self.start_positions[idx]
        end = start + self.segment_length

        # Extract input signals (2 channels)
        eeg_segment = np.stack([
            self.eeg_raw[start:end],
            self.eeg_norm[start:end]
        ], axis=0)

        # Extract target signal
        target_segment = self.rpeak_targets[start:end]

        return torch.FloatTensor(eeg_segment), torch.FloatTensor(target_segment).unsqueeze(0)

def process_signal_for_unet(file_path):
    """Process signal for U-Net training"""
    print(f"🔬 Processing {file_path} for U-Net...")

    # Load data
    data = pd.read_csv(file_path, skiprows=5, header=None)
    # CORRECTED: Column 1 is real ECG, Column 2 is EEG for prediction
    ecg = data.iloc[:, 1].values.astype(float)  # Real ECG signal
    eeg = data.iloc[:, 2].values.astype(float)  # EEG signal for prediction

    # Remove NaN values
    valid_indices = ~(np.isnan(ecg) | np.isnan(eeg))
    ecg, eeg = ecg[valid_indices], eeg[valid_indices]
    print(f"  📊 Loaded {len(ecg)} samples at 250Hz")

    # Apply filtering
    nyquist = 0.5 * 250

    # 60Hz notch filter
    b_notch, a_notch = signal.butter(2, [58/nyquist, 62/nyquist], btype='bandstop')
    eeg_notched = signal.filtfilt(b_notch, a_notch, eeg)

    # Bandpass filter (0.5-40Hz)
    low, high = 0.5 / nyquist, 40.0 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    eeg_filtered = signal.filtfilt(b, a, eeg_notched)

    # R-peak detection on ECG
    ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=250)
    _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=250)
    rpeak_locations = rpeaks['ECG_R_Peaks']

    print(f"  💓 Found {len(rpeak_locations)} R-peaks at 250Hz")

    # Downsample to 125Hz
    eeg_raw = eeg_filtered[::2]
    rpeak_locations_downsampled = rpeak_locations // 2

    print(f"  📉 Downsampled to {len(eeg_raw)} samples at 125Hz")
    print(f"  💓 R-peaks at 125Hz: {len(rpeak_locations_downsampled)}")

    # Normalize EEG
    scaler = StandardScaler()
    eeg_normalized = scaler.fit_transform(eeg_raw.reshape(-1, 1)).flatten()

    return eeg_raw, eeg_normalized, rpeak_locations_downsampled

def create_continuous_rpeak_targets(signal_length, rpeak_locations, sigma=5):
    """
    Create continuous R-peak target signal with Gaussian peaks

    Args:
        signal_length: Length of the signal
        rpeak_locations: Array of R-peak sample indices
        sigma: Standard deviation for Gaussian peaks (in samples)

    Returns:
        Continuous target signal with Gaussian peaks at R-peak locations
    """
    targets = np.zeros(signal_length)

    # Create Gaussian peaks at each R-peak location
    for rpeak_idx in rpeak_locations:
        if 0 <= rpeak_idx < signal_length:
            # Create Gaussian centered at R-peak
            start_idx = max(0, rpeak_idx - 3*sigma)
            end_idx = min(signal_length, rpeak_idx + 3*sigma + 1)

            for i in range(start_idx, end_idx):
                # Gaussian formula: exp(-0.5 * ((x - mu) / sigma)^2)
                gaussian_value = np.exp(-0.5 * ((i - rpeak_idx) / sigma) ** 2)
                targets[i] = max(targets[i], gaussian_value)  # Take maximum for overlapping peaks

    print(f"  🎯 Created continuous targets with {len(rpeak_locations)} Gaussian peaks (sigma={sigma})")
    print(f"      Target signal range: [{targets.min():.3f}, {targets.max():.3f}]")
    print(f"      Non-zero samples: {np.sum(targets > 0.01)} ({np.sum(targets > 0.01)/len(targets)*100:.2f}%)")

    return targets

def detect_peaks_from_continuous(continuous_signal, height=0.3, distance=40):
    """
    Detect R-peaks from continuous prediction signal

    Args:
        continuous_signal: Continuous prediction from U-Net
        height: Minimum peak height threshold
        distance: Minimum distance between peaks (in samples, ~320ms at 125Hz for max ~187 BPM)

    Returns:
        Array of detected peak indices
    """
    peaks, properties = find_peaks(continuous_signal, height=height, distance=distance)
    return peaks, properties

def evaluate_rpeak_detection(predicted_peaks, true_peaks, tolerance=6):
    """
    Evaluate R-peak detection performance with one-to-one matching

    Args:
        predicted_peaks: Array of predicted peak indices
        true_peaks: Array of true R-peak indices
        tolerance: Tolerance window in samples (~50ms at 125Hz)

    Returns:
        Dictionary with precision, recall, F1-score
    """
    if len(true_peaks) == 0:
        return {
            'precision': 1.0 if len(predicted_peaks) == 0 else 0.0,
            'recall': 1.0,
            'f1': 1.0 if len(predicted_peaks) == 0 else 0.0,
            'true_positives': 0,
            'false_positives': len(predicted_peaks),
            'false_negatives': 0
        }

    if len(predicted_peaks) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': len(true_peaks)
        }

    # ONE-TO-ONE MATCHING: Each true peak can only be matched once
    matched_true_peaks = set()
    true_positives = 0
    false_positives = 0

    # Check each predicted peak
    for pred_peak in predicted_peaks:
        # Look for unmatched true peak within tolerance window
        matched = False

        for true_peak in true_peaks:
            if true_peak in matched_true_peaks:
                continue  # This true peak already matched

            if abs(pred_peak - true_peak) <= tolerance:
                true_positives += 1
                matched_true_peaks.add(true_peak)
                matched = True
                break

        if not matched:
            false_positives += 1

    false_negatives = len(true_peaks) - true_positives

    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

def map_rpeaks_to_segment(rpeak_locations, segment_start, segment_length):
    """
    Map global R-peak locations to segment-relative positions

    Args:
        rpeak_locations: Array of global R-peak indices
        segment_start: Start index of the segment in global signal
        segment_length: Length of the segment

    Returns:
        Array of R-peak indices relative to segment start
    """
    segment_end = segment_start + segment_length

    # Find R-peaks within this segment
    segment_rpeaks = []
    for rpeak_idx in rpeak_locations:
        if segment_start <= rpeak_idx < segment_end:
            relative_idx = rpeak_idx - segment_start
            segment_rpeaks.append(relative_idx)

    return np.array(segment_rpeaks)

def train_unet_model(train_loader, val_loader, config):
    """Train U-Net model for R-peak detection"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  🖥️  Using device: {device}")

    # Model
    model = RPeakUNet(n_channels=2, n_classes=1).to(device)

    # Loss function (MSE for continuous targets)
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(),
                           lr=config['learning_rate'],
                           weight_decay=config['weight_decay'])

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    # Training metrics
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None

    print("🎓 Starting U-Net training...")

    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        num_train_batches = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_train_loss += loss.item()
            num_train_batches += 1

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item()
                num_val_batches += 1

        scheduler.step()

        avg_train_loss = epoch_train_loss / num_train_batches
        avg_val_loss = epoch_val_loss / num_val_batches

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()

        print(f"Epoch {epoch+1:2d}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, best_val_loss

def time_based_split_sequences(dataset, test_size=0.2):
    """Time-based split for sequence dataset"""
    total_samples = len(dataset)
    split_idx = int(total_samples * (1 - test_size))

    # Split dataset indices
    train_indices = list(range(split_idx))
    test_indices = list(range(split_idx, total_samples))

    return train_indices, test_indices

def main():
    """Main U-Net training pipeline"""
    print("🚀 ===== U-NET R-PEAK DETECTION =====")
    print("Sequence-to-sequence approach with continuous targets")
    print("=" * 50)

    config = {
        'segment_length': 250,    # 2 seconds at 125Hz
        'stride': 125,           # 1 second stride (50% overlap)
        'gaussian_sigma': 5,     # ~40ms peak width at 125Hz
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'epochs': 20,
        'batch_size': 16
    }

    file_path = "OpenBCI-RAW-2025-09-14_12-26-20.txt"
    if not os.path.exists(file_path):
        print(f"❌ ERROR: {file_path} not found!")
        return

    # Process signals
    eeg_raw, eeg_norm, rpeak_locations = process_signal_for_unet(file_path)

    if len(rpeak_locations) == 0:
        print("❌ ERROR: No R-peaks found!")
        return

    # Create continuous R-peak targets
    rpeak_targets = create_continuous_rpeak_targets(
        len(eeg_raw), rpeak_locations, sigma=config['gaussian_sigma']
    )

    # Create dataset
    dataset = RPeakSequenceDataset(
        eeg_raw, eeg_norm, rpeak_targets,
        segment_length=config['segment_length'],
        stride=config['stride']
    )

    # Time-based split (80% train, 20% validation)
    train_indices, val_indices = time_based_split_sequences(dataset, test_size=0.2)

    print(f"\n📊 TIME-BASED SPLIT:")
    print(f"   Train segments: {len(train_indices)}")
    print(f"   Validation segments: {len(val_indices)}")

    # Create data loaders
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False)

    # Train model
    model, train_losses, val_losses, best_val_loss = train_unet_model(train_loader, val_loader, config)

    # Evaluation on validation set with ground truth R-peak locations
    print(f"\n🔍 EVALUATING ON VALIDATION SET...")
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Predict on validation segments
    all_predictions = []
    val_segment_starts = []

    # Get the actual validation segment start positions
    for idx in val_indices:
        val_segment_starts.append(dataset.start_positions[idx])

    with torch.no_grad():
        for data, targets in val_loader:
            data = data.to(device)
            outputs = model(data)
            all_predictions.append(outputs.cpu().numpy())

    # Concatenate all predictions
    predictions = np.concatenate(all_predictions, axis=0)

    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"   Evaluating ALL {len(predictions)} validation segments...")

    # Evaluate ALL validation segments using ground truth R-peak locations
    segment_metrics = []
    segments_with_rpeaks = 0

    for i in range(len(predictions)):
        pred_signal = predictions[i, 0]  # Shape: [segment_length]

        # Detect peaks in prediction with physiological constraints
        detected_peaks, _ = detect_peaks_from_continuous(pred_signal, height=0.3, distance=40)

        # Get ground truth R-peaks for this segment
        segment_start = val_segment_starts[i]
        true_peaks = map_rpeaks_to_segment(rpeak_locations, segment_start, config['segment_length'])

        # Evaluate only segments that have R-peaks
        if len(true_peaks) > 0:
            segments_with_rpeaks += 1
            metrics = evaluate_rpeak_detection(detected_peaks, true_peaks, tolerance=6)
            segment_metrics.append(metrics)

            if i < 20:  # Print first 20 for debugging
                print(f"   Segment {i}: {len(true_peaks)} true peaks, {len(detected_peaks)} detected, F1={metrics['f1']:.3f}")

    print(f"\n   📊 Evaluation Summary:")
    print(f"       Total validation segments: {len(predictions)}")
    print(f"       Segments with R-peaks: {segments_with_rpeaks}")
    print(f"       Evaluation coverage: {segments_with_rpeaks/len(predictions)*100:.1f}%")

    # Calculate average metrics
    if segment_metrics:
        avg_precision = np.mean([m['precision'] for m in segment_metrics])
        avg_recall = np.mean([m['recall'] for m in segment_metrics])
        avg_f1 = np.mean([m['f1'] for m in segment_metrics])

        print(f"\n🏆 AVERAGE VALIDATION METRICS (on segments with R-peaks):")
        print(f"   Precision: {avg_precision:.4f}")
        print(f"   Recall: {avg_recall:.4f}")
        print(f"   F1-Score: {avg_f1:.4f}")
        print(f"   Best Validation Loss: {best_val_loss:.6f}")

    # Save model and results
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/logs', exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save model
    torch.save(model.state_dict(), f'outputs/models/unet_rpeak_model_{timestamp}.pth')

    # Save results
    results = {
        'timestamp': timestamp,
        'config': config,
        'best_val_loss': float(best_val_loss),
        'avg_precision': float(avg_precision) if segment_metrics else 0.0,
        'avg_recall': float(avg_recall) if segment_metrics else 0.0,
        'avg_f1': float(avg_f1) if segment_metrics else 0.0,
        'num_validation_segments': len(segment_metrics),
        'approach': 'U-Net sequence-to-sequence'
    }

    with open(f'outputs/logs/unet_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Create training curves plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('U-Net Training Curves')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    if segment_metrics:
        f1_scores = [m['f1'] for m in segment_metrics]
        plt.hist(f1_scores, bins=10, alpha=0.7, edgecolor='black')
        plt.title(f'F1-Score Distribution (Avg: {avg_f1:.3f})')
        plt.xlabel('F1-Score')
        plt.ylabel('Number of Segments')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'outputs/plots/unet_training_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n💾 Model saved to outputs/models/unet_rpeak_model_{timestamp}.pth")
    print(f"📄 Results saved to outputs/logs/unet_results_{timestamp}.json")
    print(f"📊 Training curves saved to outputs/plots/unet_training_analysis_{timestamp}.png")

    # Compare with previous classification approach
    previous_best = 0.185  # From TimesNet classification
    if segment_metrics and avg_f1 > previous_best:
        improvement = (avg_f1 - previous_best) / previous_best * 100
        print(f"\n🎉 U-Net approach achieved {improvement:.1f}% improvement over classification!")

    return avg_f1 if segment_metrics else 0.0, results

if __name__ == "__main__":
    best_f1, results = main()
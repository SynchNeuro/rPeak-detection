#!/usr/bin/env python3
"""
Quick fix to evaluate U-Net with adaptive threshold
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import find_peaks
import sys
import os

# Add parent directory to path to import from unet_rpeak_detection
sys.path.append('.')
from src.unet_rpeak_detection import *

def evaluate_with_adaptive_threshold(model_path, data_file):
    """Evaluate U-Net with adaptive threshold finding"""

    # Load data
    eeg_raw, eeg_norm, rpeak_locations = process_signal_for_unet(data_file)

    config = {
        'segment_length': 250,
        'stride': 125,
        'gaussian_sigma': 5,
    }

    # Create targets
    rpeak_targets = create_continuous_rpeak_targets(
        len(eeg_raw), rpeak_locations, sigma=config['gaussian_sigma']
    )

    # Create dataset
    dataset = RPeakSequenceDataset(
        eeg_raw, eeg_norm, rpeak_targets,
        segment_length=config['segment_length'],
        stride=config['stride']
    )

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RPeakUNet(n_channels=2, n_classes=1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Time-based split to get validation indices
    train_indices, val_indices = time_based_split_sequences(dataset, test_size=0.2)

    print(f"🔍 ADAPTIVE THRESHOLD EVALUATION")
    print(f"   Validation segments: {len(val_indices)}")

    # Get validation predictions
    all_predictions = []
    val_segment_starts = []

    for idx in val_indices[:100]:  # Test on first 100 validation segments
        val_segment_starts.append(dataset.start_positions[idx])

        # Get segment data
        segment_data, _ = dataset[idx]

        with torch.no_grad():
            segment_data = segment_data.unsqueeze(0).to(device)
            output = model(segment_data)
            all_predictions.append(output.cpu().numpy()[0, 0])

    predictions = np.array(all_predictions)
    print(f"   Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")

    # Try different thresholds
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25]

    print(f"\n📊 THRESHOLD ANALYSIS:")
    best_f1 = 0
    best_threshold = 0.1

    for threshold in thresholds:
        segment_metrics = []

        for i, pred_signal in enumerate(predictions):
            # Detect peaks with current threshold
            detected_peaks, _ = find_peaks(pred_signal, height=threshold, distance=40)

            # Get ground truth R-peaks for this segment
            segment_start = val_segment_starts[i]
            true_peaks = map_rpeaks_to_segment(rpeak_locations, segment_start, config['segment_length'])

            if len(true_peaks) > 0:  # Only evaluate segments with R-peaks
                metrics = evaluate_rpeak_detection(detected_peaks, true_peaks, tolerance=6)
                segment_metrics.append(metrics)

        if segment_metrics:
            avg_f1 = np.mean([m['f1'] for m in segment_metrics])
            avg_precision = np.mean([m['precision'] for m in segment_metrics])
            avg_recall = np.mean([m['recall'] for m in segment_metrics])

            print(f"   Threshold {threshold:.2f}: F1={avg_f1:.4f}, P={avg_precision:.4f}, R={avg_recall:.4f} ({len(segment_metrics)} segments)")

            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_threshold = threshold

    print(f"\n🏆 BEST RESULTS:")
    print(f"   Best threshold: {best_threshold}")
    print(f"   Best F1-score: {best_f1:.4f}")

    return best_f1, best_threshold

if __name__ == "__main__":
    model_path = "outputs/models/unet_rpeak_model_20250919_182002.pth"
    data_file = "OpenBCI-RAW-2025-09-14_12-26-20.txt"

    if os.path.exists(model_path):
        best_f1, best_threshold = evaluate_with_adaptive_threshold(model_path, data_file)
    else:
        print(f"Model not found: {model_path}")
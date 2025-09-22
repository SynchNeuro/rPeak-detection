#!/usr/bin/env python3
"""
Debug R-Peak Alignment
======================

Check if R-peak indices actually correspond to peaks in the EEG signal.
Examine signal values at R-peak locations vs nearby values.

Author: Claude Code
Date: September 19, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import neurokit2 as nk
from scipy import signal
import os

def debug_rpeak_alignment():
    """Debug R-peak alignment by checking signal values"""
    print("🔍 ===== DEBUGGING R-PEAK ALIGNMENT =====")
    print("Checking if R-peak indices correspond to actual peaks in EEG signal")
    print("=" * 60)

    # Load and process data
    print("📊 Loading and processing data...")
    data = pd.read_csv("OpenBCI-RAW-2025-09-14_12-26-20.txt", skiprows=5, header=None)
    ecg_raw = data.iloc[:, 0].values.astype(float)
    eeg_raw = data.iloc[:, 1].values.astype(float)

    # Remove NaN
    valid_indices = ~(np.isnan(ecg_raw) | np.isnan(eeg_raw))
    ecg_raw, eeg_raw = ecg_raw[valid_indices], eeg_raw[valid_indices]

    print(f"  📈 Loaded {len(ecg_raw)} samples at 250Hz")

    # Apply EEG filtering
    print("🔧 Applying EEG filtering...")
    nyquist = 0.5 * 250

    # 60Hz notch filter
    b_notch, a_notch = signal.butter(2, [58/nyquist, 62/nyquist], btype='bandstop')
    eeg_notched = signal.filtfilt(b_notch, a_notch, eeg_raw)

    # Bandpass filter (0.5-40Hz)
    low, high = 0.5 / nyquist, 40.0 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    eeg_filtered = signal.filtfilt(b, a, eeg_notched)

    # R-peak detection on ECG
    print("💓 Detecting R-peaks from ECG...")
    ecg_cleaned = nk.ecg_clean(ecg_raw, sampling_rate=250)
    _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=250)
    rpeak_locations_250hz = rpeaks['ECG_R_Peaks']

    print(f"  🔍 Found {len(rpeak_locations_250hz)} R-peaks in ECG")

    # Downsample to 125Hz
    print("📉 Downsampling to 125Hz...")
    eeg_filtered_125hz = eeg_filtered[::2]

    # Convert R-peak indices to 125Hz
    rpeak_locations_125hz = rpeak_locations_250hz // 2

    print(f"  📊 EEG at 125Hz: {len(eeg_filtered_125hz)} samples")
    print(f"  💓 R-peaks at 125Hz: {len(rpeak_locations_125hz)} locations")

    # Check alignment for first 10 R-peaks
    print(f"\n🔍 ALIGNMENT CHECK - First 10 R-peaks:")
    print(f"{'R-peak#':<8} {'Index':<8} {'EEG Value':<12} {'Local Max':<12} {'Offset':<8} {'Is Peak?':<10}")
    print("-" * 70)

    alignment_issues = 0
    local_maxima_found = 0

    for i, rpeak_idx in enumerate(rpeak_locations_125hz[:10]):
        if 0 <= rpeak_idx < len(eeg_filtered_125hz):
            # Get EEG value at R-peak location
            rpeak_value = eeg_filtered_125hz[rpeak_idx]

            # Check local neighborhood (±10 samples = ±80ms at 125Hz)
            window_size = 10
            start_idx = max(0, rpeak_idx - window_size)
            end_idx = min(len(eeg_filtered_125hz), rpeak_idx + window_size + 1)

            local_window = eeg_filtered_125hz[start_idx:end_idx]
            local_max_value = np.max(local_window)
            local_max_idx = start_idx + np.argmax(local_window)

            # Calculate offset from R-peak to actual local maximum
            offset = local_max_idx - rpeak_idx
            offset_ms = offset * (1000 / 125)  # Convert to milliseconds

            # Check if R-peak is actually the local maximum
            is_peak = (rpeak_value == local_max_value)

            if not is_peak:
                alignment_issues += 1
            else:
                local_maxima_found += 1

            print(f"{i+1:<8} {rpeak_idx:<8} {rpeak_value:<12.3f} {local_max_value:<12.3f} {offset_ms:<8.1f}ms {str(is_peak):<10}")

    print("-" * 70)
    print(f"📊 Alignment Summary:")
    print(f"   R-peaks that are local maxima: {local_maxima_found}/10 ({local_maxima_found/10*100:.1f}%)")
    print(f"   R-peaks with alignment issues: {alignment_issues}/10 ({alignment_issues/10*100:.1f}%)")

    # Detailed analysis for first 5 R-peaks
    print(f"\n🔬 DETAILED ANALYSIS - First 5 R-peaks:")

    fig, axes = plt.subplots(5, 1, figsize=(15, 20))

    for i, rpeak_idx in enumerate(rpeak_locations_125hz[:5]):
        if i >= 5:
            break

        ax = axes[i]

        # Show wider window around R-peak (±50 samples = ±400ms)
        window_size = 50
        start_idx = max(0, rpeak_idx - window_size)
        end_idx = min(len(eeg_filtered_125hz), rpeak_idx + window_size + 1)

        window_indices = np.arange(start_idx, end_idx)
        window_values = eeg_filtered_125hz[start_idx:end_idx]
        time_ms = (window_indices - rpeak_idx) * (1000 / 125)  # Time relative to R-peak in ms

        # Plot EEG signal
        ax.plot(time_ms, window_values, 'b-', linewidth=1, alpha=0.8, label='EEG Signal')

        # Mark R-peak location
        rpeak_value = eeg_filtered_125hz[rpeak_idx]
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='R-peak from ECG')
        ax.scatter([0], [rpeak_value], color='red', s=100, zorder=5, marker='v')

        # Find and mark actual local maximum in the window
        local_max_idx = start_idx + np.argmax(window_values)
        local_max_time = (local_max_idx - rpeak_idx) * (1000 / 125)
        local_max_value = eeg_filtered_125hz[local_max_idx]

        ax.axvline(x=local_max_time, color='green', linestyle=':', linewidth=2, label='EEG Local Max')
        ax.scatter([local_max_time], [local_max_value], color='green', s=100, zorder=5, marker='^')

        # Find and mark local minimum
        local_min_idx = start_idx + np.argmin(window_values)
        local_min_time = (local_min_idx - rpeak_idx) * (1000 / 125)
        local_min_value = eeg_filtered_125hz[local_min_idx]

        ax.scatter([local_min_time], [local_min_value], color='purple', s=100, zorder=5, marker='v')

        # Calculate timing offset
        timing_offset = local_max_time

        ax.set_title(f'R-peak #{i+1}: Index {rpeak_idx}, Timing offset: {timing_offset:.1f}ms', fontsize=12)
        ax.set_xlabel('Time relative to R-peak (ms)')
        ax.set_ylabel('EEG Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add text box with values
        textstr = f'R-peak value: {rpeak_value:.3f}\nLocal max: {local_max_value:.3f}\nOffset: {timing_offset:.1f}ms'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig('outputs/plots/rpeak_alignment_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Statistical analysis of all R-peaks
    print(f"\n📈 STATISTICAL ANALYSIS - All {len(rpeak_locations_125hz)} R-peaks:")

    all_offsets = []
    peak_alignments = []

    for rpeak_idx in rpeak_locations_125hz:
        if 10 <= rpeak_idx < len(eeg_filtered_125hz) - 10:  # Ensure we have room for window
            # Check local neighborhood (±10 samples)
            window_size = 10
            start_idx = rpeak_idx - window_size
            end_idx = rpeak_idx + window_size + 1

            local_window = eeg_filtered_125hz[start_idx:end_idx]
            local_max_idx = start_idx + np.argmax(local_window)

            # Calculate offset
            offset = local_max_idx - rpeak_idx
            offset_ms = offset * (1000 / 125)
            all_offsets.append(offset_ms)

            # Check if R-peak is the local maximum
            rpeak_value = eeg_filtered_125hz[rpeak_idx]
            local_max_value = np.max(local_window)
            is_peak = (rpeak_value == local_max_value)
            peak_alignments.append(is_peak)

    if all_offsets:
        mean_offset = np.mean(all_offsets)
        std_offset = np.std(all_offsets)
        median_offset = np.median(all_offsets)
        alignment_rate = np.mean(peak_alignments) * 100

        print(f"   📊 Timing Offset Statistics:")
        print(f"      Mean offset: {mean_offset:.1f} ± {std_offset:.1f} ms")
        print(f"      Median offset: {median_offset:.1f} ms")
        print(f"      Range: {np.min(all_offsets):.1f} to {np.max(all_offsets):.1f} ms")
        print(f"   📍 Alignment Statistics:")
        print(f"      R-peaks that are local EEG maxima: {alignment_rate:.1f}%")

        # Create histogram of offsets
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.hist(all_offsets, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(mean_offset, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_offset:.1f}ms')
        plt.axvline(median_offset, color='green', linestyle='--', linewidth=2, label=f'Median: {median_offset:.1f}ms')
        plt.xlabel('Timing Offset (ms)')
        plt.ylabel('Count')
        plt.title('Distribution of R-peak Timing Offsets')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        alignment_counts = [np.sum(peak_alignments), len(peak_alignments) - np.sum(peak_alignments)]
        plt.pie(alignment_counts, labels=['Aligned', 'Misaligned'], autopct='%1.1f%%',
                colors=['lightgreen', 'lightcoral'])
        plt.title('R-peak Alignment Rate')

        plt.subplot(2, 2, 3)
        plt.scatter(range(len(all_offsets)), all_offsets, alpha=0.6, s=20)
        plt.axhline(mean_offset, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_offset:.1f}ms')
        plt.xlabel('R-peak Number')
        plt.ylabel('Timing Offset (ms)')
        plt.title('Timing Offset Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 4)
        # Show offset vs EEG amplitude at R-peak
        rpeak_amplitudes = []
        for rpeak_idx in rpeak_locations_125hz:
            if 10 <= rpeak_idx < len(eeg_filtered_125hz) - 10:
                rpeak_amplitudes.append(eeg_filtered_125hz[rpeak_idx])

        if len(rpeak_amplitudes) == len(all_offsets):
            plt.scatter(rpeak_amplitudes, all_offsets, alpha=0.6, s=20)
            plt.xlabel('EEG Amplitude at R-peak')
            plt.ylabel('Timing Offset (ms)')
            plt.title('Offset vs EEG Amplitude')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('outputs/plots/rpeak_offset_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Determine if there's a systematic timing issue
        if abs(mean_offset) > 20:  # More than 20ms average offset
            print(f"\n⚠️  TIMING ISSUE DETECTED!")
            print(f"   Average offset of {mean_offset:.1f}ms suggests systematic misalignment")
            print(f"   This could explain the 200ms delay observation you mentioned")

            if mean_offset > 0:
                print(f"   R-peaks are consistently LATE by {mean_offset:.1f}ms")
                print(f"   EEG peaks occur BEFORE the detected R-peak locations")
            else:
                print(f"   R-peaks are consistently EARLY by {abs(mean_offset):.1f}ms")
                print(f"   EEG peaks occur AFTER the detected R-peak locations")

        elif alignment_rate < 50:
            print(f"\n⚠️  ALIGNMENT ISSUE DETECTED!")
            print(f"   Only {alignment_rate:.1f}% of R-peaks are local EEG maxima")
            print(f"   This suggests poor correlation between ECG R-peaks and EEG features")
        else:
            print(f"\n✅ Alignment appears reasonable")
            print(f"   {alignment_rate:.1f}% of R-peaks are local EEG maxima")
            print(f"   Average offset of {mean_offset:.1f}ms is within acceptable range")

    print(f"\n📊 Analysis complete!")
    print(f"📈 Detailed plots saved:")
    print(f"   outputs/plots/rpeak_alignment_analysis.png")
    print(f"   outputs/plots/rpeak_offset_statistics.png")

    return {
        'mean_offset_ms': mean_offset if all_offsets else 0,
        'alignment_rate_percent': alignment_rate if all_offsets else 0,
        'total_rpeaks_analyzed': len(all_offsets),
        'systematic_issue': abs(mean_offset) > 20 if all_offsets else False
    }

if __name__ == "__main__":
    os.makedirs('outputs/plots', exist_ok=True)
    results = debug_rpeak_alignment()
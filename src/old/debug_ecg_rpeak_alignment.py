#!/usr/bin/env python3
"""
Debug ECG R-Peak Alignment
==========================

Check if R-peak detection on ECG is correct - whether detected R-peak indices
actually correspond to the highest ECG values (peaks) in the local neighborhood.

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

def debug_ecg_rpeak_alignment():
    """Debug ECG R-peak alignment by checking ECG signal values"""
    print("🔍 ===== DEBUGGING ECG R-PEAK ALIGNMENT =====")
    print("Checking if R-peak indices correspond to actual peaks in ECG signal")
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

    # Clean ECG and detect R-peaks (same as in training)
    print("💓 Detecting R-peaks from ECG using NeuroKit2...")
    ecg_cleaned = nk.ecg_clean(ecg_raw, sampling_rate=250)
    _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=250)
    rpeak_locations = rpeaks['ECG_R_Peaks']

    print(f"  🔍 Found {len(rpeak_locations)} R-peaks in ECG")

    # Check alignment for first 10 R-peaks
    print(f"\n🔍 ECG ALIGNMENT CHECK - First 10 R-peaks:")
    print(f"{'R-peak#':<8} {'Index':<8} {'ECG Value':<12} {'Local Max':<12} {'Offset':<8} {'Is Peak?':<10}")
    print("-" * 70)

    alignment_issues = 0
    local_maxima_found = 0

    for i, rpeak_idx in enumerate(rpeak_locations[:10]):
        if 0 <= rpeak_idx < len(ecg_cleaned):
            # Get ECG value at R-peak location
            rpeak_value = ecg_cleaned[rpeak_idx]

            # Check local neighborhood (±25 samples = ±100ms at 250Hz)
            window_size = 25
            start_idx = max(0, rpeak_idx - window_size)
            end_idx = min(len(ecg_cleaned), rpeak_idx + window_size + 1)

            local_window = ecg_cleaned[start_idx:end_idx]
            local_max_value = np.max(local_window)
            local_max_idx = start_idx + np.argmax(local_window)

            # Calculate offset from R-peak to actual local maximum
            offset = local_max_idx - rpeak_idx
            offset_ms = offset * (1000 / 250)  # Convert to milliseconds

            # Check if R-peak is actually the local maximum
            is_peak = (rpeak_value == local_max_value)

            if not is_peak:
                alignment_issues += 1
            else:
                local_maxima_found += 1

            print(f"{i+1:<8} {rpeak_idx:<8} {rpeak_value:<12.3f} {local_max_value:<12.3f} {offset_ms:<8.1f}ms {str(is_peak):<10}")

    print("-" * 70)
    print(f"📊 ECG Alignment Summary:")
    print(f"   R-peaks that are local ECG maxima: {local_maxima_found}/10 ({local_maxima_found/10*100:.1f}%)")
    print(f"   R-peaks with alignment issues: {alignment_issues}/10 ({alignment_issues/10*100:.1f}%)")

    # Detailed analysis for first 5 R-peaks
    print(f"\n🔬 DETAILED ECG ANALYSIS - First 5 R-peaks:")

    fig, axes = plt.subplots(5, 1, figsize=(15, 20))

    for i, rpeak_idx in enumerate(rpeak_locations[:5]):
        if i >= 5:
            break

        ax = axes[i]

        # Show wider window around R-peak (±125 samples = ±500ms at 250Hz)
        window_size = 125
        start_idx = max(0, rpeak_idx - window_size)
        end_idx = min(len(ecg_cleaned), rpeak_idx + window_size + 1)

        window_indices = np.arange(start_idx, end_idx)
        window_values = ecg_cleaned[start_idx:end_idx]
        time_ms = (window_indices - rpeak_idx) * (1000 / 250)  # Time relative to R-peak in ms

        # Plot ECG signal
        ax.plot(time_ms, window_values, 'r-', linewidth=1, alpha=0.8, label='Cleaned ECG')

        # Mark detected R-peak location
        rpeak_value = ecg_cleaned[rpeak_idx]
        ax.axvline(x=0, color='blue', linestyle='--', linewidth=2, label='Detected R-peak')
        ax.scatter([0], [rpeak_value], color='blue', s=100, zorder=5, marker='v')

        # Find and mark actual local maximum in the window
        local_max_idx = start_idx + np.argmax(window_values)
        local_max_time = (local_max_idx - rpeak_idx) * (1000 / 250)
        local_max_value = ecg_cleaned[local_max_idx]

        ax.axvline(x=local_max_time, color='green', linestyle=':', linewidth=2, label='True ECG Max')
        ax.scatter([local_max_time], [local_max_value], color='green', s=100, zorder=5, marker='^')

        # Find and mark local minimum
        local_min_idx = start_idx + np.argmin(window_values)
        local_min_time = (local_min_idx - rpeak_idx) * (1000 / 250)
        local_min_value = ecg_cleaned[local_min_idx]

        ax.scatter([local_min_time], [local_min_value], color='purple', s=100, zorder=5, marker='v')

        # Calculate timing offset
        timing_offset = local_max_time

        ax.set_title(f'ECG R-peak #{i+1}: Index {rpeak_idx}, Timing offset: {timing_offset:.1f}ms', fontsize=12)
        ax.set_xlabel('Time relative to detected R-peak (ms)')
        ax.set_ylabel('ECG Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add text box with values
        textstr = f'Detected R-peak: {rpeak_value:.3f}\nTrue ECG max: {local_max_value:.3f}\nOffset: {timing_offset:.1f}ms'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig('outputs/plots/ecg_rpeak_alignment_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Statistical analysis of all R-peaks
    print(f"\n📈 STATISTICAL ANALYSIS - All {len(rpeak_locations)} ECG R-peaks:")

    all_offsets = []
    peak_alignments = []
    offset_values = []

    for rpeak_idx in rpeak_locations:
        if 25 <= rpeak_idx < len(ecg_cleaned) - 25:  # Ensure we have room for window
            # Check local neighborhood (±25 samples = ±100ms at 250Hz)
            window_size = 25
            start_idx = rpeak_idx - window_size
            end_idx = rpeak_idx + window_size + 1

            local_window = ecg_cleaned[start_idx:end_idx]
            local_max_idx = start_idx + np.argmax(local_window)

            # Calculate offset
            offset = local_max_idx - rpeak_idx
            offset_ms = offset * (1000 / 250)
            all_offsets.append(offset_ms)
            offset_values.append(offset)

            # Check if R-peak is the local maximum
            rpeak_value = ecg_cleaned[rpeak_idx]
            local_max_value = np.max(local_window)
            is_peak = abs(rpeak_value - local_max_value) < 1e-6  # Account for floating point precision
            peak_alignments.append(is_peak)

    if all_offsets:
        mean_offset = np.mean(all_offsets)
        std_offset = np.std(all_offsets)
        median_offset = np.median(all_offsets)
        alignment_rate = np.mean(peak_alignments) * 100

        print(f"   📊 ECG Timing Offset Statistics:")
        print(f"      Mean offset: {mean_offset:.1f} ± {std_offset:.1f} ms")
        print(f"      Median offset: {median_offset:.1f} ms")
        print(f"      Range: {np.min(all_offsets):.1f} to {np.max(all_offsets):.1f} ms")
        print(f"   📍 ECG Alignment Statistics:")
        print(f"      R-peaks that are local ECG maxima: {alignment_rate:.1f}%")

        # Check for systematic patterns
        offset_counts = np.bincount(np.array(offset_values) + 25)  # Shift by 25 to make non-negative
        most_common_offset = np.argmax(offset_counts) - 25
        most_common_offset_ms = most_common_offset * (1000 / 250)

        print(f"   📈 Most common offset: {most_common_offset} samples ({most_common_offset_ms:.1f}ms)")

        # Create histogram of offsets
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.hist(all_offsets, bins=50, alpha=0.7, color='red', edgecolor='black')
        plt.axvline(mean_offset, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_offset:.1f}ms')
        plt.axvline(median_offset, color='green', linestyle='--', linewidth=2, label=f'Median: {median_offset:.1f}ms')
        plt.axvline(0, color='black', linestyle='-', linewidth=1, label='Perfect alignment')
        plt.xlabel('ECG Timing Offset (ms)')
        plt.ylabel('Count')
        plt.title('Distribution of ECG R-peak Timing Offsets')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        alignment_counts = [np.sum(peak_alignments), len(peak_alignments) - np.sum(peak_alignments)]
        plt.pie(alignment_counts, labels=['Perfectly Aligned', 'Misaligned'], autopct='%1.1f%%',
                colors=['lightgreen', 'lightcoral'])
        plt.title('ECG R-peak Alignment Rate')

        plt.subplot(2, 2, 3)
        plt.scatter(range(len(all_offsets)), all_offsets, alpha=0.6, s=20, color='red')
        plt.axhline(mean_offset, color='blue', linestyle='--', alpha=0.7, label=f'Mean: {mean_offset:.1f}ms')
        plt.axhline(0, color='black', linestyle='-', alpha=0.7, label='Perfect alignment')
        plt.xlabel('R-peak Number')
        plt.ylabel('ECG Timing Offset (ms)')
        plt.title('ECG Timing Offset Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 4)
        # Show distribution by offset value
        unique_offsets, counts = np.unique(offset_values, return_counts=True)
        offset_ms_unique = unique_offsets * (1000 / 250)
        plt.bar(offset_ms_unique, counts, alpha=0.7, color='orange')
        plt.xlabel('Offset (ms)')
        plt.ylabel('Count')
        plt.title('R-peak Offset Distribution')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('outputs/plots/ecg_rpeak_offset_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Determine if R-peak detection is working correctly
        if alignment_rate > 80:
            print(f"\n✅ ECG R-PEAK DETECTION APPEARS CORRECT")
            print(f"   {alignment_rate:.1f}% of detected R-peaks are local ECG maxima")
            print(f"   Mean offset of {mean_offset:.1f}ms is acceptable for R-peak detection")
        elif alignment_rate > 50:
            print(f"\n⚠️  ECG R-PEAK DETECTION HAS MINOR ISSUES")
            print(f"   {alignment_rate:.1f}% of detected R-peaks are local ECG maxima")
            print(f"   Some systematic offset may be present: {mean_offset:.1f}ms")
        else:
            print(f"\n❌ ECG R-PEAK DETECTION HAS MAJOR ISSUES")
            print(f"   Only {alignment_rate:.1f}% of detected R-peaks are local ECG maxima")
            print(f"   R-peak detection algorithm may be faulty")

        if abs(mean_offset) > 10:  # More than 10ms average offset
            print(f"\n⚠️  SYSTEMATIC TIMING OFFSET DETECTED!")
            print(f"   Average offset of {mean_offset:.1f}ms suggests consistent timing bias")
            if mean_offset > 0:
                print(f"   Detected R-peaks are consistently LATE")
            else:
                print(f"   Detected R-peaks are consistently EARLY")

    print(f"\n📊 ECG R-peak analysis complete!")
    print(f"📈 Detailed plots saved:")
    print(f"   outputs/plots/ecg_rpeak_alignment_analysis.png")
    print(f"   outputs/plots/ecg_rpeak_offset_statistics.png")

    return {
        'mean_offset_ms': mean_offset if all_offsets else 0,
        'alignment_rate_percent': alignment_rate if all_offsets else 0,
        'total_rpeaks_analyzed': len(all_offsets),
        'systematic_issue': abs(mean_offset) > 10 if all_offsets else False
    }

if __name__ == "__main__":
    os.makedirs('outputs/plots', exist_ok=True)
    results = debug_ecg_rpeak_alignment()
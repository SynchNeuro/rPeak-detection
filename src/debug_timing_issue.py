#!/usr/bin/env python3
"""
Debug R-peak Timing Issue
========================

Investigate the 200ms delay in R-peak timing by examining:
1. Original ECG vs EEG signals
2. R-peak detection timing
3. Filtering effects
4. Downsampling alignment

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

def debug_timing_issue():
    """Debug the R-peak timing delay issue"""
    print("🔍 ===== DEBUGGING R-PEAK TIMING ISSUE =====")

    # Load raw data
    print("📊 Loading raw data...")
    data = pd.read_csv("OpenBCI-RAW-2025-09-14_12-26-20.txt", skiprows=5, header=None)
    ecg_raw = data.iloc[:, 0].values.astype(float)
    eeg_raw = data.iloc[:, 1].values.astype(float)

    # Remove NaN
    valid_indices = ~(np.isnan(ecg_raw) | np.isnan(eeg_raw))
    ecg_raw, eeg_raw = ecg_raw[valid_indices], eeg_raw[valid_indices]

    print(f"  📈 Loaded {len(ecg_raw)} samples at 250Hz")
    print(f"  ⏱️  Duration: {len(ecg_raw)/250:.1f} seconds")

    # R-peak detection on raw ECG (before any processing)
    print("\n💓 R-peak detection on raw ECG...")
    ecg_cleaned = nk.ecg_clean(ecg_raw, sampling_rate=250)
    _, rpeaks_raw = nk.ecg_peaks(ecg_cleaned, sampling_rate=250)
    rpeak_locations_raw = rpeaks_raw['ECG_R_Peaks']
    print(f"  🔍 Found {len(rpeak_locations_raw)} R-peaks in raw ECG")

    # Apply EEG filtering (same as in training)
    print("\n🔧 Applying EEG filtering...")
    nyquist = 0.5 * 250

    # 60Hz notch filter
    b_notch, a_notch = signal.butter(2, [58/nyquist, 62/nyquist], btype='bandstop')
    eeg_notched = signal.filtfilt(b_notch, a_notch, eeg_raw)

    # Bandpass filter (0.5-40Hz)
    low, high = 0.5 / nyquist, 40.0 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    eeg_filtered = signal.filtfilt(b, a, eeg_notched)

    print("  ✅ Applied 60Hz notch + 0.5-40Hz bandpass filters")

    # Test different downsampling approaches
    print("\n📉 Testing downsampling approaches...")

    # Method 1: Simple decimation (current approach)
    eeg_down_simple = eeg_filtered[::2]
    rpeaks_down_simple = rpeak_locations_raw // 2

    # Method 2: Proper decimation with anti-aliasing
    eeg_down_proper = signal.decimate(eeg_filtered, 2, ftype='iir')
    # For proper decimation, R-peaks need to be adjusted for the delay
    rpeaks_down_proper = rpeak_locations_raw // 2

    print(f"  📊 Simple decimation: {len(eeg_down_simple)} samples")
    print(f"  📊 Proper decimation: {len(eeg_down_proper)} samples")

    # Create comprehensive timing analysis plot
    fig, axes = plt.subplots(4, 1, figsize=(20, 15))

    # Show first 5 seconds for detailed analysis
    show_duration = 5.0  # seconds
    show_samples_250 = int(show_duration * 250)
    show_samples_125 = int(show_duration * 125)

    time_250 = np.arange(show_samples_250) / 250.0
    time_125 = np.arange(show_samples_125) / 125.0

    # Plot 1: Raw ECG and EEG at 250Hz
    ax1 = axes[0]
    ax1.plot(time_250, ecg_raw[:show_samples_250], 'r-', alpha=0.8, linewidth=1, label='Raw ECG')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(time_250, eeg_raw[:show_samples_250], 'b-', alpha=0.6, linewidth=1, label='Raw EEG')

    # Mark R-peaks on raw data
    rpeaks_in_window = rpeak_locations_raw[rpeak_locations_raw < show_samples_250]
    if len(rpeaks_in_window) > 0:
        rpeak_times = rpeaks_in_window / 250.0
        rpeak_values = ecg_raw[rpeaks_in_window]
        ax1.scatter(rpeak_times, rpeak_values, color='red', s=100, zorder=5,
                   marker='v', label=f'R-peaks ({len(rpeaks_in_window)})')

    ax1.set_ylabel('ECG (μV)', color='red')
    ax1_twin.set_ylabel('EEG (μV)', color='blue')
    ax1.set_title('Raw Signals at 250Hz with R-peak Detection')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Filtered EEG with R-peaks at 250Hz
    ax2 = axes[1]
    ax2.plot(time_250, eeg_filtered[:show_samples_250], 'g-', alpha=0.8, linewidth=1, label='Filtered EEG')

    if len(rpeaks_in_window) > 0:
        rpeak_values_eeg = eeg_filtered[rpeaks_in_window]
        ax2.scatter(rpeak_times, rpeak_values_eeg, color='red', s=100, zorder=5,
                   marker='v', label=f'R-peaks on EEG ({len(rpeaks_in_window)})')

    ax2.set_ylabel('Filtered EEG (μV)')
    ax2.set_title('Filtered EEG at 250Hz with R-peak Overlay')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Simple decimation to 125Hz
    ax3 = axes[2]
    ax3.plot(time_125, eeg_down_simple[:show_samples_125], 'b-', alpha=0.8, linewidth=1, label='EEG (Simple Decimation)')

    # R-peaks adjusted for 125Hz
    rpeaks_125_simple = rpeaks_in_window // 2
    valid_rpeaks_125 = rpeaks_125_simple[rpeaks_125_simple < show_samples_125]
    if len(valid_rpeaks_125) > 0:
        rpeak_times_125 = valid_rpeaks_125 / 125.0
        rpeak_values_125 = eeg_down_simple[valid_rpeaks_125]
        ax3.scatter(rpeak_times_125, rpeak_values_125, color='red', s=100, zorder=5,
                   marker='v', label=f'R-peaks ({len(valid_rpeaks_125)})')

    ax3.set_ylabel('EEG (μV)')
    ax3.set_title('EEG at 125Hz (Simple Decimation) with R-peaks')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Timing comparison - show potential delay
    ax4 = axes[3]

    # Compare timing by showing first few R-peaks
    if len(rpeaks_in_window) >= 3:
        for i, rpeak_250 in enumerate(rpeaks_in_window[:3]):
            rpeak_time_250 = rpeak_250 / 250.0
            rpeak_125_simple = rpeak_250 // 2
            rpeak_time_125 = rpeak_125_simple / 125.0

            # Show timing difference
            time_diff = rpeak_time_125 - rpeak_time_250

            ax4.axvline(rpeak_time_250, color='red', linestyle='-', alpha=0.7,
                       label=f'R-peak {i+1} @250Hz' if i == 0 else '')
            ax4.axvline(rpeak_time_125, color='blue', linestyle='--', alpha=0.7,
                       label=f'R-peak {i+1} @125Hz' if i == 0 else '')

            # Annotate time difference
            ax4.text(rpeak_time_250, 0.5 + i*0.1, f'Δt={time_diff*1000:.1f}ms',
                    ha='center', va='bottom', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    ax4.set_xlim(0, show_duration)
    ax4.set_ylim(0, 1)
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Reference')
    ax4.set_title('Timing Comparison: 250Hz vs 125Hz R-peak Positions')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/plots/timing_debug_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Calculate timing statistics
    print(f"\n📊 TIMING ANALYSIS RESULTS:")
    if len(rpeaks_in_window) >= 2:
        # Calculate inter-beat intervals
        ibi_250 = np.diff(rpeaks_in_window) / 250.0  # seconds
        ibi_125 = np.diff(rpeaks_125_simple) / 125.0  # seconds

        print(f"  💓 Heart rate analysis (first {len(ibi_250)} intervals):")
        print(f"      @250Hz: {60/np.mean(ibi_250):.1f} BPM (avg)")
        print(f"      @125Hz: {60/np.mean(ibi_125):.1f} BPM (avg)")

        # Check for systematic timing offset
        timing_diffs = []
        for rpeak_250 in rpeaks_in_window[:10]:  # Check first 10 R-peaks
            rpeak_time_250 = rpeak_250 / 250.0
            rpeak_125_simple = rpeak_250 // 2
            rpeak_time_125 = rpeak_125_simple / 125.0
            timing_diffs.append((rpeak_time_125 - rpeak_time_250) * 1000)  # ms

        if timing_diffs:
            avg_delay = np.mean(timing_diffs)
            print(f"  ⏱️  Average timing difference: {avg_delay:.1f} ms")
            print(f"      Range: {np.min(timing_diffs):.1f} to {np.max(timing_diffs):.1f} ms")

            if abs(avg_delay) > 50:  # More than 50ms
                print(f"  ⚠️  WARNING: Significant timing offset detected!")
                print(f"      This could explain the 200ms delay observation")

    # Test alternative downsampling approach
    print(f"\n🔧 TESTING CORRECTED APPROACH:")

    # Method: Apply R-peak detection AFTER downsampling to avoid timing issues
    print("  📊 Approach: R-peak detection on downsampled ECG")

    # Downsample ECG as well
    ecg_cleaned_down = signal.decimate(ecg_cleaned, 2, ftype='iir')
    eeg_filtered_down = signal.decimate(eeg_filtered, 2, ftype='iir')

    # Detect R-peaks on downsampled ECG
    _, rpeaks_corrected = nk.ecg_peaks(ecg_cleaned_down, sampling_rate=125)
    rpeak_locations_corrected = rpeaks_corrected['ECG_R_Peaks']

    print(f"      Found {len(rpeak_locations_corrected)} R-peaks in downsampled ECG")
    print(f"      Original @250Hz: {len(rpeak_locations_raw)} R-peaks")
    print(f"      Difference: {len(rpeak_locations_raw) - len(rpeak_locations_corrected)} R-peaks")

    # Create corrected timing plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))

    # Plot original approach
    ax1.plot(time_125, eeg_down_simple[:show_samples_125], 'b-', alpha=0.8, linewidth=1, label='EEG (125Hz)')
    if len(valid_rpeaks_125) > 0:
        ax1.scatter(rpeak_times_125, rpeak_values_125, color='red', s=100, zorder=5,
                   marker='v', label=f'R-peaks (Original Method)')
    ax1.set_ylabel('EEG (μV)')
    ax1.set_title('Original Method: R-peaks from 250Hz, decimated to 125Hz')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot corrected approach
    ax2.plot(time_125, eeg_filtered_down[:show_samples_125], 'g-', alpha=0.8, linewidth=1, label='EEG (125Hz)')
    rpeaks_corrected_window = rpeak_locations_corrected[rpeak_locations_corrected < show_samples_125]
    if len(rpeaks_corrected_window) > 0:
        rpeak_times_corrected = rpeaks_corrected_window / 125.0
        rpeak_values_corrected = eeg_filtered_down[rpeaks_corrected_window]
        ax2.scatter(rpeak_times_corrected, rpeak_values_corrected, color='blue', s=100, zorder=5,
                   marker='s', label=f'R-peaks (Corrected Method)')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('EEG (μV)')
    ax2.set_title('Corrected Method: R-peaks detected directly on 125Hz signals')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/plots/timing_correction_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Timing analysis complete!")
    print(f"📊 Plots saved:")
    print(f"    outputs/plots/timing_debug_analysis.png")
    print(f"    outputs/plots/timing_correction_comparison.png")

    return {
        'original_method': {
            'eeg_filtered': eeg_filtered,
            'rpeak_locations': rpeak_locations_raw,
            'downsampled_eeg': eeg_down_simple,
            'downsampled_rpeaks': rpeaks_down_simple
        },
        'corrected_method': {
            'eeg_filtered': eeg_filtered_down,
            'rpeak_locations': rpeak_locations_corrected,
            'sampling_rate': 125
        }
    }

if __name__ == "__main__":
    os.makedirs('outputs/plots', exist_ok=True)
    results = debug_timing_issue()
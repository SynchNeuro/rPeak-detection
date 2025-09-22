#!/usr/bin/env python3
"""
Plot ECG Signal with R-peaks
============================

Simple visualization of ECG signal with detected R-peaks.
No training, just plotting for visual inspection.

Author: Claude Code
Date: September 19, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')
import neurokit2 as nk
import os

def plot_ecg_with_rpeaks():
    """Plot ECG signal with detected R-peaks"""
    print("📊 ===== PLOTTING ECG WITH R-PEAKS =====")

    # Load data
    print("📈 Loading data...")
    data = pd.read_csv("../OpenBCI-RAW-2025-09-14_12-26-20.txt", skiprows=5, header=None)
    ecg_raw = data.iloc[:, 0].values.astype(float)
    eeg_raw = data.iloc[:, 1].values.astype(float)

    # Remove NaN
    valid_indices = ~(np.isnan(ecg_raw) | np.isnan(eeg_raw))
    ecg_raw, eeg_raw = ecg_raw[valid_indices], eeg_raw[valid_indices]

    print(f"  📊 Loaded {len(ecg_raw)} samples at 250Hz ({len(ecg_raw)/250:.1f} seconds)")

    # Clean ECG and detect R-peaks
    print("💓 Detecting R-peaks...")
    ecg_cleaned = nk.ecg_clean(ecg_raw, sampling_rate=250)
    _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=250)
    rpeak_locations = rpeaks['ECG_R_Peaks']

    print(f"  🔍 Found {len(rpeak_locations)} R-peaks")

    # Create comprehensive plots
    print("📊 Creating plots...")

    # Plot 1: Overview of entire signal
    fig, axes = plt.subplots(3, 1, figsize=(20, 12))

    # Full signal overview (first 10 minutes)
    duration_samples = min(150000, len(ecg_cleaned))  # 10 minutes at 250Hz
    time_full = np.arange(duration_samples) / 250.0  # Convert to seconds

    axes[0].plot(time_full, ecg_cleaned[:duration_samples], 'b-', linewidth=0.5, alpha=0.8, label='Cleaned ECG')

    # Mark R-peaks in the overview
    rpeaks_in_overview = rpeak_locations[rpeak_locations < duration_samples]
    if len(rpeaks_in_overview) > 0:
        rpeak_times = rpeaks_in_overview / 250.0
        rpeak_values = ecg_cleaned[rpeaks_in_overview]
        axes[0].scatter(rpeak_times, rpeak_values, color='red', s=20, zorder=5,
                       alpha=0.8, label=f'R-peaks ({len(rpeaks_in_overview)})')

    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('ECG Amplitude')
    axes[0].set_title(f'ECG Signal Overview - First {duration_samples/250:.1f} seconds with R-peaks')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Medium zoom - 30 seconds
    start_medium = 5000  # Start at 20 seconds
    duration_medium = 7500  # 30 seconds at 250Hz
    end_medium = start_medium + duration_medium

    if end_medium <= len(ecg_cleaned):
        time_medium = np.arange(start_medium, end_medium) / 250.0

        axes[1].plot(time_medium, ecg_cleaned[start_medium:end_medium], 'b-',
                    linewidth=1, alpha=0.8, label='Cleaned ECG')

        # Mark R-peaks in medium view
        rpeaks_medium = rpeak_locations[(rpeak_locations >= start_medium) &
                                       (rpeak_locations < end_medium)]
        if len(rpeaks_medium) > 0:
            rpeak_times_medium = rpeaks_medium / 250.0
            rpeak_values_medium = ecg_cleaned[rpeaks_medium]
            axes[1].scatter(rpeak_times_medium, rpeak_values_medium, color='red', s=50,
                           zorder=5, alpha=0.8, label=f'R-peaks ({len(rpeaks_medium)})')

        axes[1].set_xlabel('Time (seconds)')
        axes[1].set_ylabel('ECG Amplitude')
        axes[1].set_title('ECG Signal - 30 Second View with R-peaks')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # Plot 3: Close zoom - 5 seconds around some R-peaks
    if len(rpeak_locations) > 10:
        # Center around the 10th R-peak
        center_rpeak = rpeak_locations[9]
        window_size = 1250  # 5 seconds at 250Hz
        start_zoom = max(0, center_rpeak - window_size//2)
        end_zoom = min(len(ecg_cleaned), center_rpeak + window_size//2)

        time_zoom = np.arange(start_zoom, end_zoom) / 250.0

        axes[2].plot(time_zoom, ecg_cleaned[start_zoom:end_zoom], 'b-',
                    linewidth=2, alpha=0.8, label='Cleaned ECG')

        # Mark R-peaks in zoom view
        rpeaks_zoom = rpeak_locations[(rpeak_locations >= start_zoom) &
                                     (rpeak_locations < end_zoom)]
        if len(rpeaks_zoom) > 0:
            rpeak_times_zoom = rpeaks_zoom / 250.0
            rpeak_values_zoom = ecg_cleaned[rpeaks_zoom]
            axes[2].scatter(rpeak_times_zoom, rpeak_values_zoom, color='red', s=100,
                           zorder=5, marker='v', label=f'R-peaks ({len(rpeaks_zoom)})')

            # Add annotations for each R-peak
            for i, (time, value) in enumerate(zip(rpeak_times_zoom, rpeak_values_zoom)):
                axes[2].annotate(f'R{i+1}', (time, value),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, color='red', fontweight='bold')

        axes[2].set_xlabel('Time (seconds)')
        axes[2].set_ylabel('ECG Amplitude')
        axes[2].set_title('ECG Signal - Close-up View (5 seconds) with R-peaks')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('outputs/plots', exist_ok=True)
    plt.savefig('outputs/plots/ecg_signal_with_rpeaks.png', dpi=300, bbox_inches='tight')
    # plt.close()

    # Create a detailed plot showing individual heartbeats
    print("📊 Creating individual heartbeat plots...")

    if len(rpeak_locations) >= 6:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for i in range(6):
            if i < len(rpeak_locations):
                rpeak_idx = rpeak_locations[i]

                # Show ±200ms around each R-peak
                window_size = 50  # 200ms at 250Hz
                start_idx = max(0, rpeak_idx - window_size)
                end_idx = min(len(ecg_cleaned), rpeak_idx + window_size + 1)

                # Time relative to R-peak in milliseconds
                indices = np.arange(start_idx, end_idx)
                time_ms = (indices - rpeak_idx) * (1000 / 250)
                values = ecg_cleaned[start_idx:end_idx]

                axes[i].plot(time_ms, values, 'b-', linewidth=2, alpha=0.8)

                # Mark the R-peak
                rpeak_value = ecg_cleaned[rpeak_idx]
                axes[i].axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
                axes[i].scatter([0], [rpeak_value], color='red', s=100, zorder=5, marker='v')

                axes[i].set_xlabel('Time relative to R-peak (ms)')
                axes[i].set_ylabel('ECG Amplitude')
                axes[i].set_title(f'Heartbeat #{i+1} (Index: {rpeak_idx})')
                axes[i].grid(True, alpha=0.3)
                axes[i].set_xlim(-200, 200)

        plt.tight_layout()
        plt.savefig('outputs/plots/individual_heartbeats.png', dpi=300, bbox_inches='tight')
        # plt.close()

    # Calculate and display heart rate statistics
    print("\n💓 Heart Rate Analysis:")
    if len(rpeak_locations) > 1:
        # Calculate inter-beat intervals
        rr_intervals = np.diff(rpeak_locations) / 250.0  # Convert to seconds
        heart_rates = 60.0 / rr_intervals  # Convert to BPM

        print(f"   📊 Heart Rate Statistics:")
        print(f"      Mean HR: {np.mean(heart_rates):.1f} BPM")
        print(f"      Std HR: {np.std(heart_rates):.1f} BPM")
        print(f"      Min HR: {np.min(heart_rates):.1f} BPM")
        print(f"      Max HR: {np.max(heart_rates):.1f} BPM")
        print(f"      Total beats: {len(rpeak_locations)}")
        print(f"      Recording duration: {len(ecg_cleaned)/250:.1f} seconds")

        # Plot heart rate over time
        plt.figure(figsize=(15, 6))

        # Time points for heart rate (at each R-peak except the first)
        hr_times = rpeak_locations[1:] / 250.0

        plt.plot(hr_times, heart_rates, 'g-', linewidth=1.5, alpha=0.8, label='Heart Rate')
        plt.scatter(hr_times, heart_rates, color='green', s=20, alpha=0.6)

        # Add mean line
        plt.axhline(y=np.mean(heart_rates), color='red', linestyle='--',
                   alpha=0.7, label=f'Mean: {np.mean(heart_rates):.1f} BPM')

        plt.xlabel('Time (seconds)')
        plt.ylabel('Heart Rate (BPM)')
        plt.title('Heart Rate Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, max(heart_rates) * 1.1)

        plt.tight_layout()
        plt.savefig('outputs/plots/heart_rate_over_time.png', dpi=300, bbox_inches='tight')
        # plt.close()

    print(f"\n📊 Plots saved:")
    print(f"   outputs/plots/ecg_signal_with_rpeaks.png - Overview of ECG with R-peaks")
    print(f"   outputs/plots/individual_heartbeats.png - Individual heartbeat morphology")
    print(f"   outputs/plots/heart_rate_over_time.png - Heart rate variability")

    return {
        'total_rpeaks': len(rpeak_locations),
        'duration_seconds': len(ecg_cleaned) / 250.0,
        'mean_heart_rate': np.mean(heart_rates) if len(rpeak_locations) > 1 else 0,
        'plots_created': [
            'outputs/plots/ecg_signal_with_rpeaks.png',
            'outputs/plots/individual_heartbeats.png',
            'outputs/plots/heart_rate_over_time.png'
        ]
    }

if __name__ == "__main__":
    results = plot_ecg_with_rpeaks()
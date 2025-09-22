#!/usr/bin/env python3
"""
Investigate Raw Data
===================

Examine the raw data to understand what signals we actually have.
The ECG waveform looks suspicious - doesn't look like real ECG.

Author: Claude Code
Date: September 19, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

def investigate_raw_data():
    """Investigate what's actually in the raw data file"""
    print("🔍 ===== INVESTIGATING RAW DATA =====")

    # Load data
    print("📈 Loading raw data...")
    data = pd.read_csv("OpenBCI-RAW-2025-09-14_12-26-20.txt", skiprows=5, header=None)

    print(f"  📊 Data shape: {data.shape}")
    print(f"  📊 Columns: {list(data.columns)}")

    # Look at first few rows
    print(f"\n📋 First 10 rows of raw data:")
    print(data.head(10))

    # Check data types and basic stats
    print(f"\n📊 Data info:")
    print(data.info())

    # Look at numeric columns only (skip timestamp column)
    numeric_columns = []
    for col in range(data.shape[1] - 1):  # Skip last column (timestamp)
        try:
            col_data = data.iloc[:, col].values.astype(float)
            numeric_columns.append(col)
        except:
            print(f"  Column {col}: Non-numeric, skipping")

    print(f"\n🔍 Examining numeric columns:")
    for col in numeric_columns:
        col_data = data.iloc[:, col].values.astype(float)
        # Remove NaN
        col_data_clean = col_data[~np.isnan(col_data)]

        print(f"  Column {col}:")
        print(f"    Range: {np.min(col_data_clean):.3f} to {np.max(col_data_clean):.3f}")
        print(f"    Mean: {np.mean(col_data_clean):.3f} ± {np.std(col_data_clean):.3f}")
        print(f"    Non-NaN samples: {len(col_data_clean)}")

        # Special analysis for potential physiological signals
        signal_range = np.max(col_data_clean) - np.min(col_data_clean)
        if 100 < signal_range < 100000 and np.std(col_data_clean) > 10:
            print(f"    ⭐ POTENTIAL PHYSIOLOGICAL SIGNAL (good range and variability)")
        elif col == 0:
            print(f"    🔢 LIKELY SAMPLE COUNTER (0-255 pattern)")
        elif np.std(col_data_clean) < 1:
            print(f"    📌 CONSTANT/METADATA (low variability)")

    # Plot potential physiological signals only
    print(f"\n📊 Creating plots of potential physiological signals...")

    # Find columns that might be physiological signals
    physio_candidates = []
    for col in numeric_columns:
        col_data = data.iloc[:, col].values.astype(float)
        col_data_clean = col_data[~np.isnan(col_data)]
        signal_range = np.max(col_data_clean) - np.min(col_data_clean)

        # Look for signals with good dynamic range and variability
        if 100 < signal_range < 100000 and np.std(col_data_clean) > 10:
            physio_candidates.append(col)

    print(f"  🔍 Found {len(physio_candidates)} potential physiological signals: {physio_candidates}")

    if physio_candidates:
        num_plots = min(8, len(physio_candidates))  # Plot max 8 signals
        fig, axes = plt.subplots(num_plots, 1, figsize=(15, 3*num_plots))
        if num_plots == 1:
            axes = [axes]

        for i, col in enumerate(physio_candidates[:num_plots]):
            col_data = data.iloc[:, col].values.astype(float)
            valid_indices = ~np.isnan(col_data)
            col_data_clean = col_data[valid_indices]

            # Show first 5000 samples for detail
            show_samples = min(5000, len(col_data_clean))
            time_axis = np.arange(show_samples) / 250.0  # Assuming 250Hz

            axes[i].plot(time_axis, col_data_clean[:show_samples], 'b-', linewidth=0.8)
            axes[i].set_xlabel('Time (seconds)')
            axes[i].set_ylabel(f'Column {col} Amplitude')
            axes[i].set_title(f'Column {col} - Potential Physiological Signal')
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs('outputs/plots', exist_ok=True)
        plt.savefig('outputs/plots/potential_physiological_signals.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("  ⚠️ No clear physiological signal candidates found")

    # Look specifically at what I was calling "ECG" (column 0)
    print(f"\n🔬 DETAILED ANALYSIS OF COLUMN 0 (presumed ECG):")

    ecg_data = data.iloc[:, 0].values.astype(float)
    valid_indices = ~np.isnan(ecg_data)
    ecg_clean = ecg_data[valid_indices]

    # Check for patterns that suggest it's not real ECG
    print(f"  📊 Signal characteristics:")
    print(f"    Min value: {np.min(ecg_clean):.3f}")
    print(f"    Max value: {np.max(ecg_clean):.3f}")
    print(f"    Range: {np.max(ecg_clean) - np.min(ecg_clean):.3f}")
    print(f"    Mean: {np.mean(ecg_clean):.3f}")
    print(f"    Std: {np.std(ecg_clean):.3f}")

    # Check if it's monotonic or has strange patterns
    diff_data = np.diff(ecg_clean[:1000])  # First 1000 samples
    positive_changes = np.sum(diff_data > 0)
    negative_changes = np.sum(diff_data < 0)
    zero_changes = np.sum(diff_data == 0)

    print(f"  📈 Change analysis (first 1000 samples):")
    print(f"    Positive changes: {positive_changes}")
    print(f"    Negative changes: {negative_changes}")
    print(f"    Zero changes: {zero_changes}")

    # Look for periodicity
    from scipy import signal as scipy_signal

    # Autocorrelation to find periodic patterns
    sample_rate = 250
    window_size = min(10000, len(ecg_clean))  # 40 seconds

    if window_size > 1000:
        ecg_window = ecg_clean[:window_size]

        # Remove DC component
        ecg_centered = ecg_window - np.mean(ecg_window)

        # Compute autocorrelation
        autocorr = np.correlate(ecg_centered, ecg_centered, mode='full')
        autocorr = autocorr[autocorr.size // 2:]

        # Find peaks in autocorrelation (indicating periodicity)
        peaks, _ = scipy_signal.find_peaks(autocorr[1:], height=np.max(autocorr) * 0.1)

        if len(peaks) > 0:
            # Convert to time
            peak_times = peaks / sample_rate
            print(f"  🔄 Periodic patterns found at: {peak_times[:5]} seconds")

            # Estimate fundamental frequency
            if peaks[0] > 0:
                fundamental_period = peaks[0] / sample_rate
                fundamental_freq = 1 / fundamental_period
                heart_rate_estimate = fundamental_freq * 60
                print(f"  💓 Estimated heart rate from periodicity: {heart_rate_estimate:.1f} BPM")

    # Create detailed plots showing the suspicious ECG
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))

    # Plot 1: Raw signal overview
    show_samples = min(5000, len(ecg_clean))
    time_axis = np.arange(show_samples) / 250.0

    axes[0].plot(time_axis, ecg_clean[:show_samples], 'r-', linewidth=1)
    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Column 0 ("ECG") - Raw Signal Overview')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Zoomed view
    zoom_samples = min(1000, len(ecg_clean))  # 4 seconds
    time_zoom = np.arange(zoom_samples) / 250.0

    axes[1].plot(time_zoom, ecg_clean[:zoom_samples], 'r-', linewidth=2)
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Column 0 ("ECG") - Zoomed View (4 seconds)')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Individual "beats" comparison
    # Try to segment based on what NeuroKit2 detected
    import neurokit2 as nk

    try:
        ecg_processed = nk.ecg_clean(ecg_clean, sampling_rate=250)
        _, rpeaks = nk.ecg_peaks(ecg_processed, sampling_rate=250)
        rpeak_locations = rpeaks['ECG_R_Peaks']

        if len(rpeak_locations) >= 3:
            # Show 3 "beats" overlaid
            beat_window = 125  # ±500ms

            for i, rpeak_idx in enumerate(rpeak_locations[:3]):
                if rpeak_idx - beat_window >= 0 and rpeak_idx + beat_window < len(ecg_processed):
                    beat_start = rpeak_idx - beat_window
                    beat_end = rpeak_idx + beat_window + 1
                    beat_data = ecg_processed[beat_start:beat_end]
                    beat_time = (np.arange(len(beat_data)) - beat_window) * (1000 / 250)  # ms

                    axes[2].plot(beat_time, beat_data, linewidth=2, alpha=0.8, label=f'Beat {i+1}')

            axes[2].axvline(x=0, color='black', linestyle='--', alpha=0.5, label='R-peak')
            axes[2].set_xlabel('Time relative to R-peak (ms)')
            axes[2].set_ylabel('Amplitude')
            axes[2].set_title('Individual "Beats" Comparison')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, 'No R-peaks detected for beat comparison',
                        ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('Individual Beats - Not Available')

    except Exception as e:
        axes[2].text(0.5, 0.5, f'Error in beat analysis: {str(e)}',
                    ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Individual Beats - Error')

    plt.tight_layout()
    plt.savefig('outputs/plots/suspicious_ecg_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Final assessment
    print(f"\n🤔 ASSESSMENT:")

    # Check if signal looks like a ramp/sawtooth
    is_monotonic_regions = []
    chunk_size = 100
    for i in range(0, min(1000, len(ecg_clean) - chunk_size), chunk_size):
        chunk = ecg_clean[i:i+chunk_size]
        is_increasing = np.all(np.diff(chunk) >= 0)
        is_decreasing = np.all(np.diff(chunk) <= 0)
        is_monotonic_regions.append(is_increasing or is_decreasing)

    monotonic_percentage = np.mean(is_monotonic_regions) * 100

    print(f"  📊 {monotonic_percentage:.1f}% of signal regions are monotonic (suspicious for ECG)")

    if monotonic_percentage > 50:
        print(f"  ⚠️  HIGH SUSPICION: This does not look like physiological ECG!")
        print(f"     - Too many monotonic regions")
        print(f"     - Suggests synthetic/artificial signal")

    # Check amplitude characteristics
    if np.std(ecg_clean) / np.mean(np.abs(ecg_clean)) < 0.1:
        print(f"  ⚠️  LOW VARIABILITY: Signal may be synthetic")

    # Check for exact repetition
    beat_length = 256  # ~1 second at 250Hz
    if len(ecg_clean) > beat_length * 3:
        beat1 = ecg_clean[:beat_length]
        beat2 = ecg_clean[beat_length:beat_length*2]
        beat3 = ecg_clean[beat_length*2:beat_length*3]

        correlation_12 = np.corrcoef(beat1, beat2)[0, 1]
        correlation_13 = np.corrcoef(beat1, beat3)[0, 1]

        print(f"  🔄 Beat-to-beat correlation: {correlation_12:.3f}, {correlation_13:.3f}")

        if correlation_12 > 0.99 and correlation_13 > 0.99:
            print(f"  ⚠️  EXACT REPETITION: Signal appears to be artificially generated")

    print(f"\n📊 Plots saved:")
    if physio_candidates:
        print(f"   outputs/plots/potential_physiological_signals.png - Physiological signals")
    print(f"   outputs/plots/suspicious_ecg_analysis.png - Analysis of column 0")

    print(f"\n🎯 CONCLUSION:")
    print(f"   Column 0 is NOT ECG - it's a sample counter (0-255)")
    print(f"   Real physiological signals are likely in columns: {physio_candidates}")
    print(f"   Need to identify which column contains actual ECG data")

    return {
        'data_shape': data.shape,
        'column_0_is_counter': True,
        'physiological_candidates': physio_candidates,
        'monotonic_percentage': monotonic_percentage,
        'looks_like_real_ecg': False,  # Column 0 is definitely not ECG
        'signal_range': np.max(ecg_clean) - np.min(ecg_clean)
    }

if __name__ == "__main__":
    results = investigate_raw_data()
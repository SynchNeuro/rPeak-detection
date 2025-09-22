#!/usr/bin/env python3
"""
Find Real ECG Column
===================

Analyze potential ECG candidate columns to identify which one contains actual ECG data.
Test R-peak detection on each candidate to find the most realistic ECG signal.

Author: Claude Code
Date: September 19, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import neurokit2 as nk
import os

def find_real_ecg_column():
    """Test each potential ECG column to find the real one"""
    print("🔍 ===== FINDING REAL ECG COLUMN =====")

    # Load data
    print("📈 Loading data...")
    data = pd.read_csv("OpenBCI-RAW-2025-09-14_12-26-20.txt", skiprows=5, header=None)
    print(f"  📊 Data shape: {data.shape}")

    # Potential ECG candidates based on previous analysis
    # Exclude Column 0 (sample counter) and Column 22 (timestamps)
    ecg_candidates = [1, 2, 6, 14, 16, 18]

    print(f"🧪 Testing ECG candidates: {ecg_candidates}")

    results = {}

    for col_idx in ecg_candidates:
        print(f"\n🔬 ===== TESTING COLUMN {col_idx} =====")

        # Extract signal
        signal_raw = data.iloc[:, col_idx].values.astype(float)

        # Remove NaN values
        valid_indices = ~np.isnan(signal_raw)
        signal_clean = signal_raw[valid_indices]

        print(f"  📊 Signal characteristics:")
        print(f"    Range: {np.min(signal_clean):.3f} to {np.max(signal_clean):.3f}")
        print(f"    Mean: {np.mean(signal_clean):.3f} ± {np.std(signal_clean):.3f}")
        print(f"    Samples: {len(signal_clean)}")

        # Try to detect R-peaks using NeuroKit2
        try:
            # Clean the signal for ECG processing
            ecg_cleaned = nk.ecg_clean(signal_clean, sampling_rate=250)

            # Detect R-peaks
            _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=250)
            rpeak_locations = rpeaks['ECG_R_Peaks']

            num_rpeaks = len(rpeak_locations)
            duration_seconds = len(signal_clean) / 250.0

            print(f"  💓 R-peak detection results:")
            print(f"    R-peaks found: {num_rpeaks}")
            print(f"    Duration: {duration_seconds:.1f} seconds")

            if num_rpeaks > 1:
                # Calculate heart rate statistics
                rr_intervals = np.diff(rpeak_locations) / 250.0  # Convert to seconds
                heart_rates = 60.0 / rr_intervals  # Convert to BPM

                mean_hr = np.mean(heart_rates)
                std_hr = np.std(heart_rates)
                min_hr = np.min(heart_rates)
                max_hr = np.max(heart_rates)

                print(f"    Mean HR: {mean_hr:.1f} BPM")
                print(f"    HR range: {min_hr:.1f} - {max_hr:.1f} BPM")
                print(f"    HR std: {std_hr:.1f} BPM")

                # Assess realism of heart rate
                realistic_hr = 40 <= mean_hr <= 120 and std_hr < 30
                reasonable_rpeaks = 20 <= num_rpeaks <= 200  # For ~32 minutes of data

                print(f"    💡 Assessment:")
                print(f"      Realistic HR: {'✓' if realistic_hr else '✗'}")
                print(f"      Reasonable R-peak count: {'✓' if reasonable_rpeaks else '✗'}")

                # Store results
                results[col_idx] = {
                    'success': True,
                    'num_rpeaks': num_rpeaks,
                    'mean_hr': mean_hr,
                    'std_hr': std_hr,
                    'min_hr': min_hr,
                    'max_hr': max_hr,
                    'realistic_hr': realistic_hr,
                    'reasonable_rpeaks': reasonable_rpeaks,
                    'signal_range': np.max(signal_clean) - np.min(signal_clean),
                    'signal_std': np.std(signal_clean),
                    'rpeak_locations': rpeak_locations,
                    'cleaned_signal': ecg_cleaned
                }

            else:
                print(f"    ⚠️ Too few R-peaks detected")
                results[col_idx] = {'success': False, 'reason': 'too_few_rpeaks'}

        except Exception as e:
            print(f"    ❌ Error in R-peak detection: {str(e)}")
            results[col_idx] = {'success': False, 'reason': f'error: {str(e)}'}

    # Analyze results and find best candidate
    print(f"\n🏆 ===== RESULTS SUMMARY =====")

    successful_candidates = {k: v for k, v in results.items() if v.get('success', False)}

    if not successful_candidates:
        print("❌ No successful ECG candidates found!")
        return None

    print(f"✅ Successful candidates: {list(successful_candidates.keys())}")

    # Score each candidate
    best_candidate = None
    best_score = -1

    for col_idx, result in successful_candidates.items():
        # Scoring criteria:
        # 1. Realistic heart rate (40-120 BPM)
        # 2. Low heart rate variability (std < 30)
        # 3. Reasonable number of R-peaks
        # 4. Good signal variability (not too flat)

        score = 0

        if result['realistic_hr']:
            score += 50
        if result['reasonable_rpeaks']:
            score += 30
        if result['std_hr'] < 20:  # Low HR variability
            score += 20
        if result['signal_std'] > 100:  # Good signal variability
            score += 10

        # Penalty for extreme HR
        if result['mean_hr'] < 30 or result['mean_hr'] > 150:
            score -= 30

        print(f"  Column {col_idx}: Score = {score}")
        print(f"    HR: {result['mean_hr']:.1f} ± {result['std_hr']:.1f} BPM")
        print(f"    R-peaks: {result['num_rpeaks']}")
        print(f"    Signal std: {result['signal_std']:.1f}")

        if score > best_score:
            best_score = score
            best_candidate = col_idx

    if best_candidate is not None:
        print(f"\n🎯 BEST ECG CANDIDATE: Column {best_candidate}")
        best_result = results[best_candidate]
        print(f"   Mean HR: {best_result['mean_hr']:.1f} ± {best_result['std_hr']:.1f} BPM")
        print(f"   R-peaks: {best_result['num_rpeaks']}")
        print(f"   Score: {best_score}")

        # Create detailed visualization of the best candidate
        create_ecg_validation_plot(best_candidate, best_result, data)

        return best_candidate
    else:
        print("❌ Could not determine best ECG candidate")
        return None

def create_ecg_validation_plot(col_idx, result, data):
    """Create detailed validation plot for the best ECG candidate"""
    print(f"\n📊 Creating validation plot for Column {col_idx}...")

    signal_raw = data.iloc[:, col_idx].values.astype(float)
    valid_indices = ~np.isnan(signal_raw)
    signal_clean = signal_raw[valid_indices]

    ecg_cleaned = result['cleaned_signal']
    rpeak_locations = result['rpeak_locations']

    # Create comprehensive validation plot
    fig, axes = plt.subplots(4, 1, figsize=(20, 16))

    # Plot 1: Full signal overview
    duration_samples = min(75000, len(ecg_cleaned))  # 5 minutes
    time_full = np.arange(duration_samples) / 250.0

    axes[0].plot(time_full, ecg_cleaned[:duration_samples], 'b-', linewidth=0.5, alpha=0.8)

    # Mark R-peaks in overview
    rpeaks_in_overview = rpeak_locations[rpeak_locations < duration_samples]
    if len(rpeaks_in_overview) > 0:
        rpeak_times = rpeaks_in_overview / 250.0
        rpeak_values = ecg_cleaned[rpeaks_in_overview]
        axes[0].scatter(rpeak_times, rpeak_values, color='red', s=15, zorder=5, alpha=0.8,
                       label=f'R-peaks ({len(rpeaks_in_overview)})')

    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('ECG Amplitude')
    axes[0].set_title(f'Column {col_idx} - ECG Signal Overview (5 minutes)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Medium zoom - 30 seconds
    if len(rpeak_locations) > 5:
        # Center around 5th R-peak
        center_rpeak = rpeak_locations[4]
        window_size = 7500  # 30 seconds
        start_idx = max(0, center_rpeak - window_size//2)
        end_idx = min(len(ecg_cleaned), center_rpeak + window_size//2)

        time_medium = np.arange(start_idx, end_idx) / 250.0

        axes[1].plot(time_medium, ecg_cleaned[start_idx:end_idx], 'b-', linewidth=1, alpha=0.8)

        # Mark R-peaks in medium view
        rpeaks_medium = rpeak_locations[(rpeak_locations >= start_idx) &
                                       (rpeak_locations < end_idx)]
        if len(rpeaks_medium) > 0:
            rpeak_times_medium = rpeaks_medium / 250.0
            rpeak_values_medium = ecg_cleaned[rpeaks_medium]
            axes[1].scatter(rpeak_times_medium, rpeak_values_medium, color='red', s=50,
                           zorder=5, alpha=0.8, label=f'R-peaks ({len(rpeaks_medium)})')

        axes[1].set_xlabel('Time (seconds)')
        axes[1].set_ylabel('ECG Amplitude')
        axes[1].set_title('ECG Signal - 30 Second Detail View')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # Plot 3: Individual heartbeats overlay
    if len(rpeak_locations) >= 10:
        beat_window = 125  # ±500ms around R-peak

        for i, rpeak_idx in enumerate(rpeak_locations[:10]):
            if rpeak_idx - beat_window >= 0 and rpeak_idx + beat_window < len(ecg_cleaned):
                beat_start = rpeak_idx - beat_window
                beat_end = rpeak_idx + beat_window + 1
                beat_data = ecg_cleaned[beat_start:beat_end]
                beat_time = (np.arange(len(beat_data)) - beat_window) * (1000 / 250)  # ms

                axes[2].plot(beat_time, beat_data, linewidth=1.5, alpha=0.7,
                           label=f'Beat {i+1}' if i < 5 else '')

        axes[2].axvline(x=0, color='black', linestyle='--', alpha=0.5, label='R-peak')
        axes[2].set_xlabel('Time relative to R-peak (ms)')
        axes[2].set_ylabel('ECG Amplitude')
        axes[2].set_title('Individual Heartbeats Overlay (First 10 beats)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    # Plot 4: Heart rate over time
    if len(rpeak_locations) > 1:
        rr_intervals = np.diff(rpeak_locations) / 250.0
        heart_rates = 60.0 / rr_intervals
        hr_times = rpeak_locations[1:] / 250.0

        axes[3].plot(hr_times, heart_rates, 'g-', linewidth=1.5, alpha=0.8)
        axes[3].scatter(hr_times, heart_rates, color='green', s=20, alpha=0.6)

        # Add mean line
        mean_hr = np.mean(heart_rates)
        axes[3].axhline(y=mean_hr, color='red', linestyle='--', alpha=0.7,
                       label=f'Mean: {mean_hr:.1f} BPM')

        axes[3].set_xlabel('Time (seconds)')
        axes[3].set_ylabel('Heart Rate (BPM)')
        axes[3].set_title('Heart Rate Over Time')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        axes[3].set_ylim(0, max(heart_rates) * 1.1)

    plt.tight_layout()
    os.makedirs('outputs/plots', exist_ok=True)
    plt.savefig(f'outputs/plots/real_ecg_column_{col_idx}_validation.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  📊 Validation plot saved: outputs/plots/real_ecg_column_{col_idx}_validation.png")

if __name__ == "__main__":
    best_ecg_column = find_real_ecg_column()

    if best_ecg_column is not None:
        print(f"\n✅ SUCCESS: Real ECG data found in Column {best_ecg_column}")
        print(f"📝 Next steps:")
        print(f"   1. Update all R-peak detection scripts to use Column {best_ecg_column}")
        print(f"   2. Re-run experiments with correct ECG data")
        print(f"   3. Compare EEG columns (likely Column 1 or 2) for R-peak prediction")
    else:
        print(f"\n❌ FAILED: Could not identify real ECG column")
        print(f"📝 Recommendations:")
        print(f"   1. Check if ECG data is in a different format")
        print(f"   2. Try different signal processing approaches")
        print(f"   3. Verify data acquisition setup")
#!/usr/bin/env python3
"""
Debug downsampling issue - R-peaks being lost during 250Hz -> 125Hz conversion
"""

import numpy as np
import pandas as pd
import neurokit2 as nk

# Load data
file_path = "OpenBCI-RAW-2025-09-14_12-26-20.txt"
data = pd.read_csv(file_path, skiprows=5, header=None)
ecg = data.iloc[:, 0].values.astype(float)

# Remove NaN
valid_indices = ~np.isnan(ecg)
ecg = ecg[valid_indices]

print(f"Loaded {len(ecg)} ECG samples at 250Hz")

# Detect R-peaks at 250Hz
ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=250)
_, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=250)
rpeak_binary = np.zeros(len(ecg))
rpeak_locations = rpeaks['ECG_R_Peaks']
rpeak_binary[rpeak_locations] = 1

print(f"Found {len(rpeak_locations)} R-peaks at 250Hz")
print(f"R-peak positions (first 10): {rpeak_locations[:10]}")
print(f"Total R-peaks in binary signal: {np.sum(rpeak_binary)}")

# WRONG way (what we were doing):
print("\n--- WRONG DOWNSAMPLING ---")
rpeak_binary_wrong = rpeak_binary[::2]
print(f"After naive downsampling: {np.sum(rpeak_binary_wrong)} R-peaks remaining")

# CORRECT way:
print("\n--- CORRECT DOWNSAMPLING ---")
rpeak_binary_correct = np.zeros(len(ecg) // 2)
rpeak_locations_downsampled = rpeak_locations // 2

for rpeak_idx in rpeak_locations_downsampled:
    if 0 <= rpeak_idx < len(rpeak_binary_correct):
        rpeak_binary_correct[rpeak_idx] = 1

print(f"After correct downsampling: {np.sum(rpeak_binary_correct)} R-peaks remaining")
print(f"Downsampled R-peak positions (first 10): {rpeak_locations_downsampled[:10]}")

# Verify the positions make sense
print(f"\nOriginal signal length: {len(ecg)}")
print(f"Downsampled signal length: {len(rpeak_binary_correct)}")
print(f"R-peak preservation rate: {np.sum(rpeak_binary_correct) / len(rpeak_locations) * 100:.1f}%")
# R-Peak Detection Training and Validation Implementation

## Overview

This document describes the current implementation of the EEG-based R-peak detection system using deep learning. The system uses EEG signals to predict R-peaks that were originally detected from ECG signals, implementing a cross-modal signal analysis approach.

## Dataset Information

### Signal Characteristics
- **Source**: OpenBCI EXG data format (`OpenBCI-RAW-2025-09-14_12-26-20.txt`)
- **Original Length**: 485,026 samples (32.3 minutes at 250 Hz)
- **Downsampled Length**: 242,513 samples (32.3 minutes at 125 Hz)
- **Channels**:
  - Channel 0: ECG signal (ground truth for R-peak detection)
  - Channel 1: EEG signal (model input)
- **Total R-peaks**: 947 detected by neurokit2 at 125Hz
- **Heart Rate**: 58.6 BPM (realistic resting HR)

### Data Split
- **Training Set**: 169,759 samples (70%) containing 663 R-peaks
- **Validation Set**: 72,754 samples (30%) containing 284 R-peaks
- **Split Type**: Time-based chronological split (preserves temporal structure)

## Training Implementation

### Window-Based Approach
- **Window Size**: 375 samples (3.0 seconds at 125 Hz)
- **Window Strategy**: Sliding window with 1-sample stride across all training data
- **Target**: Binary classification (R-peak present/absent at window center)
- **Context**: 3-second windows capture multiple heartbeats for richer temporal context

### Positive Sample Generation
**Sampling Strategy**: Exhaustive 1-sample stride across ALL training data
```python
# Every possible window position examined
for start_idx in range(len(eeg_data) - window_size + 1):  # 1-sample stride
    center_idx = start_idx + window_size // 2

    # R-peak tolerance: ±2 samples around window center
    for offset in [-2, -1, 0, 1, 2]:
        if rpeak_binary[center_idx + offset] == 1:
            is_positive = True
```
- **Method**: **Exhaustive sampling** - every possible 375-sample window examined
- **Stride**: 1 sample (8ms at 125Hz) - maximum density
- **Tolerance**: ±2 samples (±16ms) around window center
- **Coverage**: ALL windows containing R-peaks within ±2 samples are captured
- **Multiple Heartbeats**: 3-second windows typically contain 2-3 heartbeats
- **Rich Context**: Model learns temporal patterns across multiple cardiac cycles

### Negative Sample Generation
**Sampling Strategy**: Exhaustive + Random Subsampling
```python
# Two-stage process
# Stage 1: Exhaustive candidate generation (same as positive)
for start_idx in range(len(eeg_data) - window_size + 1):  # 1-sample stride
    # Calculate minimum distance to any R-peak
    if min_distance_to_rpeak >= 5:  # 40ms minimum distance at 125Hz
        negative_candidates.append(window)

# Stage 2: Random subsampling for target ratio
target_negatives = num_positive * negative_ratio  # 10:1 ratio
negative_indices = np.random.choice(len(negative_candidates), target_negatives, replace=False)
negative_samples = [negative_candidates[i] for i in negative_indices]
```
- **Stage 1**: **Exhaustive candidate generation** - every possible 375-sample window examined
- **Filtering**: 40ms minimum distance from any R-peak at 125Hz
- **Stage 2**: **Random subsampling** to achieve 10:1 negative:positive ratio
- **Multiple Patterns**: Each 3-second negative window contains diverse non-R-peak patterns
- **Randomization**: `np.random.choice()` ensures diverse negative examples
- **No Replacement**: Each negative sample used only once

### Training Data Statistics
- **Total Training Samples**: 72,908
  - Positive: 6,628 (9.1%)
  - Negative: 66,280 (90.9%)
- **Data Utilization**: ~21% of available training data
- **Class Balance**: Much more realistic than 1:1 ratio

### Model Architecture
- **Base Model**: Enhanced TimesNet for 3-second sequences
- **Input**: 375-sample EEG windows (1 channel, 3 seconds at 125Hz)
- **Output**: Binary classification (R-peak probability)
- **Enhanced Features**:
  - d_model: 128 (increased capacity for longer sequences)
  - d_ff: 256 (increased capacity for complex temporal patterns)
  - e_layers: 3 (more layers for temporal dependencies)
  - dropout: 0.3 (increased regularization for larger model)
  - top_k: 8 (more frequency components for 3-second analysis)
  - num_kernels: 12 (more temporal kernels for complex patterns)

### Loss Function and Training
- **Loss**: Focal Loss (alpha=0.25, gamma=2.0)
  - Addresses extreme class imbalance
  - Focuses on hard examples
- **Optimizer**: AdamW (lr=0.0008, weight_decay=1e-3)
- **Scheduler**: CosineAnnealingWarmRestarts
- **Early Stopping**: Patience=7 based on validation F1-score

## Validation Implementation

The validation process uses a dual approach to evaluate both traditional metrics and clinically relevant heart rate comparison.

### Traditional Validation (Non-overlapping Windows)
```python
# Non-overlapping windows for standard metrics
stride = window_size  # 125 samples
num_windows = len(validation_data) // window_size  # 1,164 windows
```
- **Purpose**: Standard classification metrics
- **Windows**: 1,164 non-overlapping windows
- **Metrics**: Accuracy, Precision, Recall, F1-score, Specificity

### Heart Rate Validation (Sliding Windows)
```python
# Dense sliding windows for HR computation
stride = 5  # 5 samples (20ms)
num_windows = (len(validation_data) - window_size) // stride + 1  # 29,077 windows
```
- **Purpose**: Enable heart rate comparison between ECG and EEG
- **Windows**: 29,077 sliding windows with 5-sample stride
- **Rationale**: Ensures no R-peaks are missed (max 2.5 samples from nearest center)

### Peak Detection Algorithm
```python
def detect_peaks_from_probabilities(probabilities, center_indices, threshold, min_distance=50):
    # scipy.signal.find_peaks with constraints
    peaks = find_peaks(probabilities, height=threshold, distance=min_distance)
    return center_indices[peaks]
```
- **Method**: scipy.signal.find_peaks with minimum distance constraint
- **Threshold**: Optimized probability threshold (currently 0.25)
- **Min Distance**: 50 samples (200ms) between detected peaks
- **Output**: Discrete R-peak locations for HR calculation

### Heart Rate Calculation
```python
def compute_heart_rate(rpeak_indices, sampling_rate=250):
    rr_intervals = np.diff(rpeak_indices) / sampling_rate
    valid_rr = rr_intervals[(rr_intervals >= 0.3) & (rr_intervals <= 2.0)]
    hr_bpm = 60.0 / np.mean(valid_rr)
    return hr_bpm
```
- **RR Intervals**: Time differences between consecutive R-peaks
- **Outlier Removal**: RR intervals outside 0.3-2.0 seconds excluded
- **HR Calculation**: 60 / mean_RR_interval
- **Comparison**: Direct comparison of HR_ECG vs HR_EEG

## Signal Processing Pipeline

### EEG Preprocessing
1. **Data Loading**: Skip 5 header lines, handle NaN values
2. **Bandpass Filtering**: 1-35 Hz at original 250Hz (removes artifacts, focuses on neural activity)
3. **Downsampling**: Reduce from 250Hz to 125Hz by taking every other sample
4. **Standardization**: Z-score normalization applied separately to train/test

### ECG Ground Truth Generation
1. **ECG Cleaning**: neurokit2.ecg_clean() at original 250Hz for optimal accuracy
2. **R-peak Detection**: neurokit2.ecg_peaks() with 250Hz sampling rate
3. **Downsampling**: Adjust R-peak indices for 125Hz sampling rate
4. **Binary Signal**: Create binary array with 1s at R-peak locations (125Hz)

## Current Performance Results

### Traditional Metrics
- **Accuracy**: 96.9%
- **Precision**: 11.1%
- **Recall**: 9.1%
- **F1-Score**: 10.0%

### Heart Rate Comparison
- **Ground Truth R-peaks (ECG)**: 568
- **Predicted R-peaks (EEG)**: 211
- **HR from ECG**: 58.6 BPM
- **HR from EEG**: 42.7 BPM
- **HR Error**: 15.9 BPM (27.2%)

## Key Implementation Features

### Strengths
1. **Realistic Training**: 10:1 negative:positive ratio mimics real-world distribution
2. **Challenging Negatives**: 36ms minimum distance creates hard negative examples
3. **Dense Validation**: 5-sample stride ensures comprehensive R-peak detection
4. **Clinical Relevance**: Direct HR comparison enables clinical evaluation
5. **Temporal Preservation**: Time-based splits maintain realistic evaluation

### Limitations
1. **Conservative Detection**: Model detects only ~37% of actual R-peaks (211/568)
2. **HR Underestimation**: 27% error suggests missed beats
3. **Class Imbalance**: Despite improvements, still challenging for rare event detection
4. **Single Subject**: Training and validation on same subject limits generalizability

## Future Improvement Directions

1. **Threshold Optimization**: Further tune detection threshold for better recall
2. **Loss Function**: Experiment with alternative loss functions for imbalanced data
3. **Data Augmentation**: Add noise, scaling, or temporal perturbations
4. **Multi-Subject Training**: Include data from multiple subjects
5. **Ensemble Methods**: Combine multiple models for robust detection
6. **Post-processing**: Add temporal constraints and physiological constraints

## Technical Details

### File Structure
- **Main Script**: `src/best_rpeak_detection_method.py`
- **Dependencies**: TimesNet from Time-Series-Library
- **Output**: Model saved to `outputs/models/best_rpeak_detection_model.pth`

### Hardware Requirements
- **GPU**: CUDA-enabled device recommended
- **Memory**: ~8GB RAM for current dataset size
- **Training Time**: ~30-45 minutes for full dataset

### Reproducibility
- **Random Seed**: Set for numpy and torch
- **Deterministic**: Model architecture and hyperparameters fixed
- **Data Split**: Time-based split ensures consistent train/validation sets
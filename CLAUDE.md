# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an ECG/EEG signal processing project focused on R-peak detection using deep learning. The project uses EEG signals to predict R-peaks that were originally detected from ECG signals, implementing a cross-modal signal analysis approach.

## Key Commands

### Running the Scripts
```bash
python3 rpeak_detection_balanced.py    # Balanced model with focal loss and proper metrics
python3 rpeak_detection_fast.py        # Optimized model with comprehensive evaluation
python3 rpeak_detection_eeg.py         # Basic R-peak detection model
python3 debug_script.py                # Debug and test data loading
```

### Dependencies
The project requires:
- Python 3.x
- PyTorch for deep learning
- neurokit2 for ECG signal processing
- pandas, numpy for data handling
- matplotlib for visualization
- scipy for signal filtering
- scikit-learn for metrics

## Data Format

Input files are OpenBCI raw EXG data (*.txt format):
- Skip first 5 header lines
- Column 1: ECG signal (used for ground truth R-peak detection)
- Column 2: EEG signal (used as model input)
- Sampling rate: 250 Hz
- Data contains NaN values that need filtering

## Architecture

### Signal Processing Pipeline
1. **Data Loading**: Load ECG/EEG from OpenBCI format files
2. **Preprocessing**: 
   - Bandpass filter EEG (1-35 Hz)
   - Clean ECG using neurokit2
   - Standardize signals
3. **Ground Truth Generation**: Detect R-peaks from ECG using neurokit2
4. **Model Training**: Use windowed EEG data to predict R-peak locations
5. **Evaluation**: Compare predictions against ECG-derived ground truth

### Model Variants
- **rpeak_detection_balanced.py**: Uses FocalLoss, weighted sampling, comprehensive metrics (precision, recall, F1, specificity)
- **rpeak_detection_fast.py**: Optimized architecture with validation split, early stopping
- **rpeak_detection_eeg.py**: Basic CNN implementation

### Data Splitting
- Training: 65%
- Validation: 15% 
- Test: 20%
- Time-based splits (chronological order preserved)

## Key Features

### Model Architecture
- 1D CNN for temporal EEG pattern recognition
- Window size: 125 samples (0.5 seconds at 250Hz)
- Handles severe class imbalance (R-peaks are rare events)
- Batch normalization and dropout for regularization

### Evaluation Metrics
- Precision, Recall, F1-score (primary focus due to imbalance)
- Specificity, Accuracy
- AUC-ROC, AUC-PR
- Confusion matrix analysis

### Visualizations Generated
- `detailed_ecg_validation.png`: ECG with R-peaks and EEG overlay
- `comprehensive_training_curves.png`: All training metrics
- `comprehensive_results.png`: Model predictions vs ground truth

## Important Implementation Details

### Class Imbalance Handling
The dataset has severe class imbalance (~0.5% positive R-peak samples). The balanced version addresses this with:
- Focal Loss (alpha=0.25, gamma=2.0)
- Weighted random sampling
- Appropriate metrics (precision/recall over accuracy)

### Signal Processing
- EEG bandpass filtering (1-35 Hz) removes artifacts and focuses on neural activity
- ECG cleaning using neurokit2 for reliable R-peak detection
- Standardization applied separately to train/test sets

### Model Training
- Early stopping based on validation F1-score
- Learning rate scheduling
- Multiple threshold evaluation for optimal operating point
- Best model checkpointing

## Common Tasks

When working with this codebase:
1. **Adding new models**: Follow the pattern in existing files with proper Dataset and Model classes
2. **Changing preprocessing**: Modify the filtering parameters in `apply_bandpass_filter()`
3. **Adjusting evaluation**: Update threshold ranges in `evaluate_balanced_model()`
4. **Debugging data issues**: Use `debug_script.py` to test data loading and basic processing

## Generated Artifacts

The scripts generate several files:
- `best_rpeak_model.pth` / `best_balanced_model.pth`: Trained model weights
- `*.png`: Visualization plots for analysis
- Console logs with detailed metrics and training progress
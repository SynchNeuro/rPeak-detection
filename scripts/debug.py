#!/usr/bin/env python3
"""
Debug script to test data loading and basic functionality.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from src.utils.helpers import print_system_info, check_data_quality, print_data_quality_report
from src.data.loader import load_openbci_file, process_single_file, load_all_files
from src.models import get_model


def test_data_loading():
    """Test data loading functionality."""
    print("=== Testing Data Loading ===")

    try:
        # Find data files
        import glob
        files = glob.glob("*.txt")
        print(f"Found files: {files}")

        if not files:
            print("No data files found!")
            return False

        # Test loading first file
        first_file = files[0]
        print(f"\nTesting with file: {first_file}")

        # Test basic loading
        ecg, eeg = load_openbci_file(first_file, max_samples=5000)
        print(f"Loaded {len(ecg)} ECG samples, {len(eeg)} EEG samples")
        print(f"ECG range: [{np.min(ecg):.2f}, {np.max(ecg):.2f}]")
        print(f"EEG range: [{np.min(eeg):.2f}, {np.max(eeg):.2f}]")

        # Test processing with R-peak detection
        eeg_filtered, rpeak_binary, ecg_cleaned, rpeak_locations = process_single_file(first_file, max_samples=5000)
        print(f"R-peaks detected: {len(rpeak_locations)}")
        print(f"R-peak rate: {np.sum(rpeak_binary) / len(rpeak_binary) * 100:.3f}%")

        return True

    except Exception as e:
        print(f"Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_models():
    """Test model creation."""
    print("\n=== Testing Models ===")

    model_names = ['basic', 'optimized', 'improved', 'deep']

    for model_name in model_names:
        try:
            print(f"Testing {model_name} model...")
            model = get_model(model_name, input_size=125)

            # Test forward pass
            dummy_input = torch.randn(8, 125)  # Batch of 8 samples
            with torch.no_grad():
                output = model(dummy_input)

            print(f"  Parameters: {model.get_num_parameters():,}")
            print(f"  Output shape: {output.shape}")
            print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

        except Exception as e:
            print(f"  Error testing {model_name} model: {e}")


def test_preprocessing():
    """Test preprocessing functions."""
    print("\n=== Testing Preprocessing ===")

    try:
        from src.preprocessing.signal_processing import (
            apply_bandpass_filter,
            detect_r_peaks,
            standardize_signal
        )

        # Create test signal
        test_signal = np.random.randn(1000) + np.sin(np.linspace(0, 4*np.pi, 1000))
        print(f"Test signal shape: {test_signal.shape}")

        # Test bandpass filter
        filtered_signal = apply_bandpass_filter(test_signal)
        print(f"Filtered signal range: [{np.min(filtered_signal):.3f}, {np.max(filtered_signal):.3f}]")

        # Test standardization
        standardized, scaler = standardize_signal(test_signal, fit=True)
        print(f"Standardized signal mean: {np.mean(standardized):.6f}, std: {np.std(standardized):.6f}")

        print("Preprocessing tests passed!")
        return True

    except Exception as e:
        print(f"Preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_datasets():
    """Test dataset creation."""
    print("\n=== Testing Datasets ===")

    try:
        from src.data.dataset import ECGEEGDataset, BalancedECGEEGDataset
        from torch.utils.data import DataLoader

        # Create dummy data
        eeg_data = np.random.randn(1000)
        labels = np.random.choice([0, 1], size=1000, p=[0.95, 0.05])  # Imbalanced

        # Test basic dataset
        basic_dataset = ECGEEGDataset(eeg_data, labels, window_size=125)
        print(f"Basic dataset length: {len(basic_dataset)}")

        # Test balanced dataset
        balanced_dataset = BalancedECGEEGDataset(eeg_data, labels, window_size=125)
        class_dist = balanced_dataset.get_class_distribution()
        print(f"Class distribution: {class_dist}")

        # Test data loader
        loader = DataLoader(basic_dataset, batch_size=16, shuffle=True)
        sample_batch = next(iter(loader))
        print(f"Batch shapes: EEG {sample_batch[0].shape}, Labels {sample_batch[1].shape}")

        print("Dataset tests passed!")
        return True

    except Exception as e:
        print(f"Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration():
    """Test configuration system."""
    print("\n=== Testing Configuration ===")

    try:
        from src.utils.config import RPeakConfig, validate_config

        # Create default config
        config = RPeakConfig()
        print(f"Default config created")

        # Test validation
        is_valid = validate_config(config)
        print(f"Config validation: {'PASSED' if is_valid else 'FAILED'}")

        # Test config modification
        config.training.epochs = 10
        config.model.name = 'improved'
        print(f"Modified config: epochs={config.training.epochs}, model={config.model.name}")

        print("Configuration tests passed!")
        return True

    except Exception as e:
        print(f"Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main debug function."""
    print("=== R-peak Detection Debug Script ===")

    # Print system information
    print_system_info()

    # Run all tests
    tests = [
        ("Data Loading", test_data_loading),
        ("Models", test_models),
        ("Preprocessing", test_preprocessing),
        ("Datasets", test_datasets),
        ("Configuration", test_configuration)
    ]

    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False

    # Summary
    print(f"\n{'='*50}")
    print("=== Test Summary ===")
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        print(f"{test_name:<20}: {status}")

    passed_tests = sum(results.values())
    total_tests = len(results)
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("All tests passed! The system is ready for training.")
    else:
        print("Some tests failed. Please check the errors above.")

    return 0 if passed_tests == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())
"""
Configuration management for R-peak detection project.
"""

import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class DataConfig:
    """Data configuration."""
    file_pattern: str = "*.txt"
    max_samples: int = 50000
    train_ratio: float = 0.65
    val_ratio: float = 0.15
    test_ratio: float = 0.20
    window_size: int = 125
    sampling_rate: float = 250.0
    eeg_lowcut: float = 1.0
    eeg_highcut: float = 35.0
    standardize: bool = True


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "improved"
    input_size: int = 125
    dropout_rate: float = 0.3
    num_blocks: int = 4  # For deep models
    base_channels: int = 32  # For deep models


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 64
    epochs: int = 30
    early_stopping_patience: int = 10
    early_stopping_metric: str = "f1_score"
    save_best_only: bool = True

    # Loss configuration
    loss_name: str = "focal"
    loss_params: Dict[str, Any] = field(default_factory=lambda: {"alpha": 0.25, "gamma": 2.0})

    # Optimizer configuration
    optimizer_name: str = "adam"
    learning_rate: float = 0.001
    weight_decay: float = 1e-4

    # Scheduler configuration
    use_scheduler: bool = True
    scheduler_name: str = "reduce_on_plateau"
    scheduler_params: Dict[str, Any] = field(default_factory=lambda: {"patience": 5, "factor": 0.5, "verbose": True})

    # Class balancing
    use_balanced_sampler: bool = True
    use_class_weights: bool = True


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    threshold: float = 0.5
    evaluation_thresholds: list = field(default_factory=lambda: [0.3, 0.4, 0.5, 0.6, 0.7])
    find_optimal_threshold: bool = True
    optimize_for_metric: str = "f1_score"


@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    create_plots: bool = True
    ecg_validation_duration: float = 30.0
    prediction_results_duration: float = 20.0
    plot_format: str = "png"
    plot_dpi: int = 200


@dataclass
class OutputConfig:
    """Output configuration."""
    base_dir: str = "outputs"
    models_dir: str = "outputs/models"
    plots_dir: str = "outputs/plots"
    logs_dir: str = "outputs/logs"
    save_predictions: bool = True
    save_metrics: bool = True


@dataclass
class RPeakConfig:
    """Main configuration class."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Global settings
    seed: int = 42
    device: str = "auto"  # "auto", "cpu", "cuda"
    verbose: bool = True


def load_config(config_path: str) -> RPeakConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path (str): Path to configuration file

    Returns:
        RPeakConfig: Configuration object
    """
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found, using default configuration")
        return RPeakConfig()

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Create config object with nested structure
    config = RPeakConfig()

    # Update nested configurations
    if 'data' in config_dict:
        for key, value in config_dict['data'].items():
            if hasattr(config.data, key):
                setattr(config.data, key, value)

    if 'model' in config_dict:
        for key, value in config_dict['model'].items():
            if hasattr(config.model, key):
                setattr(config.model, key, value)

    if 'training' in config_dict:
        for key, value in config_dict['training'].items():
            if hasattr(config.training, key):
                setattr(config.training, key, value)

    if 'evaluation' in config_dict:
        for key, value in config_dict['evaluation'].items():
            if hasattr(config.evaluation, key):
                setattr(config.evaluation, key, value)

    if 'visualization' in config_dict:
        for key, value in config_dict['visualization'].items():
            if hasattr(config.visualization, key):
                setattr(config.visualization, key, value)

    if 'output' in config_dict:
        for key, value in config_dict['output'].items():
            if hasattr(config.output, key):
                setattr(config.output, key, value)

    # Update global settings
    for key in ['seed', 'device', 'verbose']:
        if key in config_dict:
            setattr(config, key, config_dict[key])

    return config


def save_config(config: RPeakConfig, config_path: str):
    """
    Save configuration to YAML file.

    Args:
        config (RPeakConfig): Configuration object
        config_path (str): Path to save configuration file
    """
    config_dict = asdict(config)

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def create_default_configs():
    """Create default configuration files for different model types."""
    configs_dir = "configs"
    os.makedirs(configs_dir, exist_ok=True)
    os.makedirs(os.path.join(configs_dir, "model_configs"), exist_ok=True)

    # Base configuration
    base_config = RPeakConfig()
    save_config(base_config, os.path.join(configs_dir, "base_config.yaml"))

    # CNN model configurations
    cnn_configs = {
        "basic": {"name": "basic", "dropout_rate": 0.3},
        "optimized": {"name": "optimized", "dropout_rate": 0.5},
        "improved": {"name": "improved", "dropout_rate": 0.3},
        "deep": {"name": "deep", "dropout_rate": 0.3, "num_blocks": 4, "base_channels": 32}
    }

    for model_name, model_params in cnn_configs.items():
        config = RPeakConfig()
        for key, value in model_params.items():
            setattr(config.model, key, value)

        # Adjust training parameters based on model complexity
        if model_name == "deep":
            config.training.epochs = 50
            config.training.learning_rate = 0.0005
        elif model_name == "optimized":
            config.training.batch_size = 128
            config.training.epochs = 40

        save_config(config, os.path.join(configs_dir, "model_configs", f"{model_name}_config.yaml"))

    print("Default configuration files created in 'configs/' directory")


def get_device_config(device_str: str = "auto"):
    """
    Get device configuration.

    Args:
        device_str (str): Device string ("auto", "cpu", "cuda")

    Returns:
        torch.device: PyTorch device
    """
    import torch

    if device_str == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_str == "cuda":
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("CUDA not available, falling back to CPU")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    return device


def validate_config(config: RPeakConfig) -> bool:
    """
    Validate configuration parameters.

    Args:
        config (RPeakConfig): Configuration to validate

    Returns:
        bool: True if configuration is valid
    """
    errors = []

    # Validate data ratios
    total_ratio = config.data.train_ratio + config.data.val_ratio + config.data.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        errors.append(f"Data split ratios sum to {total_ratio:.3f}, should sum to 1.0")

    # Validate data parameters
    if config.data.window_size <= 0:
        errors.append("Window size must be positive")

    if config.data.sampling_rate <= 0:
        errors.append("Sampling rate must be positive")

    # Validate training parameters
    if config.training.batch_size <= 0:
        errors.append("Batch size must be positive")

    if config.training.epochs <= 0:
        errors.append("Number of epochs must be positive")

    if config.training.learning_rate <= 0:
        errors.append("Learning rate must be positive")

    # Validate evaluation parameters
    if not (0 <= config.evaluation.threshold <= 1):
        errors.append("Evaluation threshold must be between 0 and 1")

    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False

    return True


def print_config(config: RPeakConfig):
    """
    Print configuration in a formatted way.

    Args:
        config (RPeakConfig): Configuration to print
    """
    print("=== Configuration ===")
    print(f"Data:")
    print(f"  File pattern: {config.data.file_pattern}")
    print(f"  Max samples: {config.data.max_samples}")
    print(f"  Train/Val/Test ratio: {config.data.train_ratio:.2f}/{config.data.val_ratio:.2f}/{config.data.test_ratio:.2f}")
    print(f"  Window size: {config.data.window_size}")
    print(f"  EEG filter: {config.data.eeg_lowcut}-{config.data.eeg_highcut} Hz")

    print(f"Model:")
    print(f"  Name: {config.model.name}")
    print(f"  Input size: {config.model.input_size}")
    print(f"  Dropout rate: {config.model.dropout_rate}")

    print(f"Training:")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Loss: {config.training.loss_name}")
    print(f"  Early stopping: {config.training.early_stopping_metric} (patience: {config.training.early_stopping_patience})")

    print(f"Output:")
    print(f"  Models dir: {config.output.models_dir}")
    print(f"  Plots dir: {config.output.plots_dir}")
    print()
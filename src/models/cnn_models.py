"""
CNN-based models for R-peak detection from EEG signals.
"""

import torch
import torch.nn as nn
from typing import Optional

from .base import BaseRPeakModel


class RPeakNet(BaseRPeakModel):
    """
    Basic 1D CNN for R-peak detection from EEG signals.

    This is the original simple architecture with 3 conv layers
    and 3 fully connected layers.
    """

    def __init__(self, input_size: int = 125, **kwargs):
        """
        Initialize the basic RPeakNet model.

        Args:
            input_size (int): Size of input window
            **kwargs: Additional configuration
        """
        super().__init__(input_size, **kwargs)

        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)

        # Pooling and dropout
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)

        # Calculate size after convolutions and pooling
        self.fc_input_size = 128 * (input_size // 8)  # 3 pooling operations

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Add channel dimension
        x = x.unsqueeze(1)

        # Convolutional layers
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))

        return x


class OptimizedRPeakNet(BaseRPeakModel):
    """
    Optimized 1D CNN with enhanced architecture for better performance.

    Features:
    - More convolutional layers
    - Batch normalization
    - Adaptive pooling
    - Enhanced fully connected layers
    """

    def __init__(self, input_size: int = 125, dropout_rate: float = 0.5, **kwargs):
        """
        Initialize the optimized RPeakNet model.

        Args:
            input_size (int): Size of input window
            dropout_rate (float): Dropout rate
            **kwargs: Additional configuration
        """
        super().__init__(input_size, dropout_rate=dropout_rate, **kwargs)

        # Multi-scale CNN architecture
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(256, 128, kernel_size=3, padding=1)

        # Batch normalization for better training stability
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)

        # Pooling and dropout
        self.pool = nn.MaxPool1d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(32)
        self.dropout = nn.Dropout(dropout_rate)

        # Enhanced fully connected layers
        self.fc1 = nn.Linear(128 * 32, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)

        # Activation functions
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Add channel dimension
        x = x.unsqueeze(1)

        # Convolutional layers with batch normalization
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.relu(self.bn4(self.conv4(x)))

        # Adaptive pooling to handle variable input sizes
        x = self.adaptive_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers with dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)  # Raw logits for BCEWithLogitsLoss

        return x


class ImprovedRPeakNet(BaseRPeakModel):
    """
    Improved CNN with residual-like connections and advanced architecture.

    Features:
    - Residual-like skip connections
    - Batch normalization
    - Global average pooling
    - Progressive feature expansion
    """

    def __init__(self, input_size: int = 125, dropout_rate: float = 0.3, **kwargs):
        """
        Initialize the improved RPeakNet model.

        Args:
            input_size (int): Size of input window
            dropout_rate (float): Dropout rate
            **kwargs: Additional configuration
        """
        super().__init__(input_size, dropout_rate=dropout_rate, **kwargs)

        # Residual CNN blocks
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(64, 128, kernel_size=3, padding=1)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)

        # Classifier
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        x = x.unsqueeze(1)

        # CNN with residual-like connections
        x1 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.pool(x1)

        x2 = self.relu(self.bn2(self.conv2(x1)))
        x2 = x2 + x1  # Skip connection
        x2 = self.pool(x2)

        x3 = self.relu(self.bn3(self.conv3(x2)))
        x3 = self.pool(x3)

        x4 = self.relu(self.bn4(self.conv4(x3)))
        x4 = x4 + x3  # Skip connection
        x4 = self.pool(x4)

        x5 = self.relu(self.bn5(self.conv5(x4)))
        x5 = self.global_pool(x5)

        x5 = x5.view(x5.size(0), -1)

        x = self.relu(self.fc1(x5))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # Raw logits

        return x


class ResidualBlock(nn.Module):
    """Residual block for DeepRPeakNet."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        # Skip connection projection if needed
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class DeepRPeakNet(BaseRPeakModel):
    """
    Deep CNN architecture with multiple residual blocks.

    Features:
    - Multiple residual blocks
    - Progressive feature maps growth
    - Batch normalization throughout
    - Configurable depth
    """

    def __init__(self, input_size: int = 125, num_blocks: int = 4,
                 base_channels: int = 32, dropout_rate: float = 0.3, **kwargs):
        """
        Initialize the deep RPeakNet model.

        Args:
            input_size (int): Size of input window
            num_blocks (int): Number of residual blocks
            base_channels (int): Base number of channels
            dropout_rate (float): Dropout rate
            **kwargs: Additional configuration
        """
        super().__init__(input_size, num_blocks=num_blocks,
                        base_channels=base_channels, dropout_rate=dropout_rate, **kwargs)

        self.num_blocks = num_blocks
        self.base_channels = base_channels

        # Initial convolution
        self.initial_conv = nn.Conv1d(1, base_channels, kernel_size=7, padding=3)
        self.initial_bn = nn.BatchNorm1d(base_channels)

        # Create residual blocks
        self.blocks = nn.ModuleList()
        in_channels = base_channels

        for i in range(num_blocks):
            out_channels = base_channels * (2 ** i)
            self.blocks.append(ResidualBlock(in_channels, out_channels))
            in_channels = out_channels

        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_channels // 2, 1)
        )

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with multiple residual blocks."""
        x = x.unsqueeze(1)

        # Initial convolution
        x = self.relu(self.initial_bn(self.initial_conv(x)))

        # Residual blocks (residual connection handled inside each block)
        for block in self.blocks:
            x = block(x)
            x = nn.functional.max_pool1d(x, 2)

        # Global pooling and classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


def get_model(model_name: str, **kwargs) -> BaseRPeakModel:
    """
    Factory function to get a model by name.

    Args:
        model_name (str): Name of the model
        **kwargs: Model-specific arguments

    Returns:
        BaseRPeakModel: Instantiated model

    Raises:
        ValueError: If model name is not recognized
    """
    models = {
        'basic': RPeakNet,
        'rpeak_net': RPeakNet,
        'optimized': OptimizedRPeakNet,
        'optimized_rpeak_net': OptimizedRPeakNet,
        'improved': ImprovedRPeakNet,
        'improved_rpeak_net': ImprovedRPeakNet,
        'deep': DeepRPeakNet,
        'deep_rpeak_net': DeepRPeakNet
    }

    model_name_lower = model_name.lower()
    if model_name_lower not in models:
        available_models = list(models.keys())
        raise ValueError(f"Unknown model: {model_name}. Available models: {available_models}")

    return models[model_name_lower](**kwargs)
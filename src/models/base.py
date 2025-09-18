"""
Base model class for R-peak detection models.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseRPeakModel(nn.Module, ABC):
    """
    Abstract base class for R-peak detection models.

    All R-peak detection models should inherit from this class and implement
    the required methods.
    """

    def __init__(self, input_size: int = 125, **kwargs):
        """
        Initialize the base model.

        Args:
            input_size (int): Size of input window
            **kwargs: Additional model-specific arguments
        """
        super().__init__()
        self.input_size = input_size
        self.model_config = kwargs

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Model output
        """
        pass

    def get_num_parameters(self) -> int:
        """
        Get the total number of trainable parameters.

        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            dict: Model information including architecture details
        """
        return {
            'model_name': self.__class__.__name__,
            'input_size': self.input_size,
            'num_parameters': self.get_num_parameters(),
            'config': self.model_config
        }

    def save_model(self, filepath: str):
        """
        Save model state dict.

        Args:
            filepath (str): Path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': self.model_config,
            'model_info': self.get_model_info()
        }, filepath)

    @classmethod
    def load_model(cls, filepath: str, **override_config):
        """
        Load model from file.

        Args:
            filepath (str): Path to the saved model
            **override_config: Configuration overrides

        Returns:
            BaseRPeakModel: Loaded model instance
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        config = checkpoint.get('model_config', {})
        config.update(override_config)

        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])

        return model

    def freeze_layers(self, layer_names: Optional[list] = None):
        """
        Freeze specified layers or all layers if none specified.

        Args:
            layer_names (list, optional): List of layer names to freeze
        """
        if layer_names is None:
            # Freeze all parameters
            for param in self.parameters():
                param.requires_grad = False
        else:
            # Freeze specific layers
            for name, param in self.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = False

    def unfreeze_layers(self, layer_names: Optional[list] = None):
        """
        Unfreeze specified layers or all layers if none specified.

        Args:
            layer_names (list, optional): List of layer names to unfreeze
        """
        if layer_names is None:
            # Unfreeze all parameters
            for param in self.parameters():
                param.requires_grad = True
        else:
            # Unfreeze specific layers
            for name, param in self.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = True

    def summary(self) -> str:
        """
        Get a string summary of the model.

        Returns:
            str: Model summary
        """
        info = self.get_model_info()
        summary_lines = [
            f"Model: {info['model_name']}",
            f"Input Size: {info['input_size']}",
            f"Total Parameters: {info['num_parameters']:,}",
            f"Configuration: {info['config']}"
        ]
        return "\n".join(summary_lines)
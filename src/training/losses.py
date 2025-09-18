"""
Loss functions for R-peak detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Focal Loss is designed to address class imbalance by down-weighting
    easy examples and focusing on hard examples.

    Paper: "Focal Loss for Dense Object Detection" by Lin et al.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.

        Args:
            alpha (float): Weighting factor for rare class (typically 0.25)
            gamma (float): Focusing parameter (typically 2.0)
            reduction (str): Specifies the reduction to apply to the output
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Focal Loss.

        Args:
            inputs (torch.Tensor): Raw logits from model
            targets (torch.Tensor): Ground truth labels

        Returns:
            torch.Tensor: Focal loss value
        """
        # Calculate BCE loss
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Calculate p_t
        pt = torch.exp(-ce_loss)

        # Calculate focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss for imbalanced datasets.
    """

    def __init__(self, pos_weight: float = None):
        """
        Initialize Weighted BCE Loss.

        Args:
            pos_weight (float): Weight for positive class
        """
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Weighted BCE Loss.

        Args:
            inputs (torch.Tensor): Raw logits from model
            targets (torch.Tensor): Ground truth labels

        Returns:
            torch.Tensor: Weighted BCE loss value
        """
        if self.pos_weight is not None:
            pos_weight = torch.tensor([self.pos_weight]).to(inputs.device)
            return F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=pos_weight)
        else:
            return F.binary_cross_entropy_with_logits(inputs, targets)


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation-like tasks.

    Useful for imbalanced datasets as it focuses on overlap between
    predictions and ground truth.
    """

    def __init__(self, smooth: float = 1e-8):
        """
        Initialize Dice Loss.

        Args:
            smooth (float): Smoothing factor to avoid division by zero
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Dice Loss.

        Args:
            inputs (torch.Tensor): Raw logits from model
            targets (torch.Tensor): Ground truth labels

        Returns:
            torch.Tensor: Dice loss value
        """
        # Apply sigmoid to get probabilities
        inputs = torch.sigmoid(inputs)

        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Calculate intersection and union
        intersection = (inputs * targets).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice_coeff


class CombinedLoss(nn.Module):
    """
    Combination of multiple loss functions.
    """

    def __init__(self, losses: dict):
        """
        Initialize Combined Loss.

        Args:
            losses (dict): Dictionary of {loss_name: (loss_function, weight)}
        """
        super(CombinedLoss, self).__init__()
        self.losses = losses

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Combined Loss.

        Args:
            inputs (torch.Tensor): Raw logits from model
            targets (torch.Tensor): Ground truth labels

        Returns:
            torch.Tensor: Combined loss value
        """
        total_loss = 0
        for loss_name, (loss_fn, weight) in self.losses.items():
            loss_value = loss_fn(inputs, targets)
            total_loss += weight * loss_value

        return total_loss


def get_loss_function(loss_name: str, **kwargs):
    """
    Factory function to get loss function by name.

    Args:
        loss_name (str): Name of the loss function
        **kwargs: Loss-specific arguments

    Returns:
        nn.Module: Loss function

    Raises:
        ValueError: If loss name is not recognized
    """
    losses = {
        'bce': nn.BCELoss,
        'bce_logits': nn.BCEWithLogitsLoss,
        'weighted_bce': WeightedBCELoss,
        'focal': FocalLoss,
        'dice': DiceLoss
    }

    loss_name_lower = loss_name.lower()
    if loss_name_lower not in losses:
        available_losses = list(losses.keys())
        raise ValueError(f"Unknown loss: {loss_name}. Available losses: {available_losses}")

    return losses[loss_name_lower](**kwargs)
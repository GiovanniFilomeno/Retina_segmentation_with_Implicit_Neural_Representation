"""
Loss functions for retinal vessel segmentation.

Combines Focal Loss (for handling severe class imbalance between vessel and
background pixels) with Dice Loss (for optimizing spatial overlap). This
combination is well-suited for medical image segmentation where:

- Class imbalance is extreme (vessels are ~5-15% of pixels)
- Both pixel-level accuracy and region-level overlap matter
- Hard examples (thin vessels, vessel boundaries) need extra attention

References:
    - Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
    - Milletari et al., "V-Net: Fully Convolutional Neural Networks for
      Volumetric Medical Image Segmentation" (3DV 2016)
"""

import torch
import torch.nn as nn


class BinaryFocalDiceLoss(nn.Module):
    """Combined Focal + Dice loss for binary segmentation (e.g., FIVES dataset).

    Focal Loss down-weights well-classified pixels and focuses learning on
    hard examples (misclassified vessel boundaries, thin capillaries).
    Dice Loss directly optimizes the F1/Dice coefficient, which is robust
    to class imbalance.

    Total loss = Focal_Loss + Dice_Loss

    Args:
        alpha (float): Focal loss weighting factor for the positive class.
            Higher values (e.g., 0.75) increase focus on vessel pixels.
        gamma (float): Focal loss focusing parameter. Higher values increase
            the relative loss for hard, misclassified examples.
            gamma=0 is equivalent to standard cross-entropy.
        smooth (float): Smoothing constant to avoid division by zero in Dice.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, smooth: float = 1e-5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined Focal + Dice loss.

        Args:
            inputs: Predicted probabilities of shape (N,), values in (0, 1).
            targets: Ground truth binary labels of shape (N,), values in {0, 1}.

        Returns:
            Scalar loss value.
        """
        # Numerical stability: clamp predictions away from 0 and 1
        inputs = torch.clamp(inputs, 1e-7, 1 - 1e-7)

        # --- Focal Loss ---
        # For positive pixels (target=1): -alpha * (1-p)^gamma * log(p)
        # For negative pixels (target=0): -(1-alpha) * p^gamma * log(1-p)
        focal_loss = (
            -self.alpha * (1 - inputs) ** self.gamma * targets * torch.log(inputs)
            - (1 - self.alpha) * inputs ** self.gamma * (1 - targets) * torch.log(1 - inputs)
        )
        focal_loss = focal_loss.mean()

        # --- Dice Loss ---
        # Dice = 2 * |A ∩ B| / (|A| + |B|)
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        dice_loss = 1 - (2.0 * intersection + self.smooth) / (union + self.smooth)

        return focal_loss + dice_loss


class FocalDiceLoss(nn.Module):
    """Combined Focal + Dice loss for multi-class segmentation (e.g., RAVIR dataset).

    Extends the binary version to handle K classes using one-hot encoding.
    Dice loss is computed per-class and averaged, giving equal importance to
    each class regardless of pixel frequency.

    Args:
        num_classes (int): Number of segmentation classes.
        alpha (float): Focal loss weight (default: 0.8).
        gamma (float): Focal loss focusing parameter (default: 2.0).
        smooth (float): Dice smoothing constant (default: 1e-5).
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.8,
        gamma: float = 2.0,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute multi-class Focal + Dice loss.

        Args:
            inputs: Predicted class probabilities of shape (N, num_classes).
            targets: Ground truth class indices of shape (N,) with values
                in {0, 1, ..., num_classes - 1}.

        Returns:
            Scalar loss value.
        """
        targets = targets.long()

        # Convert class indices to one-hot encoding: (N,) -> (N, K)
        targets_one_hot = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1)

        # --- Multi-class Focal Loss ---
        focal_loss = -self.alpha * (1 - inputs) ** self.gamma * targets_one_hot * torch.log(inputs + 1e-8)
        focal_loss = focal_loss.sum(dim=1).mean()

        # --- Per-class Dice Loss ---
        # Compute intersection and union per class, then average
        intersection = (inputs * targets_one_hot).sum(dim=0)  # (K,)
        union = inputs.sum(dim=0) + targets_one_hot.sum(dim=0)  # (K,)
        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_per_class.mean()

        return focal_loss + dice_loss

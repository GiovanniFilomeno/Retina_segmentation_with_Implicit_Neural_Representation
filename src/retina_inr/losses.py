"""Numerically stable focal--Dice objectives operating on raw logits."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _validate_binary_targets(targets: torch.Tensor) -> None:
    if targets.numel() == 0:
        raise ValueError("targets must not be empty")
    if not torch.isfinite(targets).all():
        raise ValueError("targets contain non-finite values")
    if not torch.all((targets == 0) | (targets == 1)):
        raise ValueError("binary targets must contain only 0 and 1")


def _validate_class_targets(targets: torch.Tensor, num_classes: int) -> None:
    if targets.numel() == 0:
        raise ValueError("targets must not be empty")
    if targets.is_floating_point():
        if not torch.isfinite(targets).all():
            raise ValueError("targets contain non-finite values")
        if not torch.equal(targets, targets.round()):
            raise ValueError("multiclass targets must be integer class indices")
    if torch.any(targets < 0) or torch.any(targets >= num_classes):
        raise ValueError(f"targets must be in [0, {num_classes - 1}]")


class BinaryFocalDiceLoss(nn.Module):
    """Combined binary focal and soft-Dice loss computed from one logit.

    ``logits`` may have shape ``(N,)``, ``(N, 1)``, or any other shape whose
    number of elements equals the target's.  The implementation uses
    :func:`torch.nn.functional.binary_cross_entropy_with_logits`, avoiding the
    unstable ``log(sigmoid(x))`` pattern.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.75,
        gamma: float = 2.0,
        smooth: float = 1e-6,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        if smooth <= 0:
            raise ValueError("smooth must be positive")
        if focal_weight < 0 or dice_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if focal_weight == 0 and dice_weight == 0:
            raise ValueError("at least one loss weight must be positive")

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.smooth = float(smooth)
        self.focal_weight = float(focal_weight)
        self.dice_weight = float(dice_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if not logits.is_floating_point():
            raise TypeError("logits must be a floating-point tensor")
        if logits.numel() != targets.numel():
            raise ValueError(
                f"logits and targets must describe the same pixels; got "
                f"{logits.numel()} and {targets.numel()} elements"
            )

        logits_flat = logits.reshape(-1)
        targets_flat = targets.to(device=logits.device, dtype=logits.dtype).reshape(-1)
        _validate_binary_targets(targets_flat)

        bce = F.binary_cross_entropy_with_logits(logits_flat, targets_flat, reduction="none")
        probability_true_class = torch.exp(-bce)
        alpha_factor = torch.where(
            targets_flat == 1,
            torch.as_tensor(self.alpha, device=logits.device, dtype=logits.dtype),
            torch.as_tensor(1.0 - self.alpha, device=logits.device, dtype=logits.dtype),
        )
        focal = (alpha_factor * (1.0 - probability_true_class).pow(self.gamma) * bce).mean()

        probabilities = torch.sigmoid(logits_flat)
        intersection = torch.sum(probabilities * targets_flat)
        denominator = torch.sum(probabilities) + torch.sum(targets_flat)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1.0 - dice

        return self.focal_weight * focal + self.dice_weight * dice_loss


class MulticlassFocalDiceLoss(nn.Module):
    """Combined multiclass focal and class-averaged soft-Dice loss from logits.

    Class channels must be last, for example ``(N, K)`` or ``(H, W, K)``.
    ``alpha`` may be omitted, a global scalar, or a sequence of K class
    weights.  The optional background exclusion affects only the Dice term.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        alpha: float | Sequence[float] | torch.Tensor | None = None,
        gamma: float = 2.0,
        smooth: float = 1e-6,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
        include_background: bool = True,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        if smooth <= 0:
            raise ValueError("smooth must be positive")
        if focal_weight < 0 or dice_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if focal_weight == 0 and dice_weight == 0:
            raise ValueError("at least one loss weight must be positive")

        class_alpha: torch.Tensor | None
        if alpha is None:
            class_alpha = None
        elif isinstance(alpha, (float, int)):
            if float(alpha) < 0:
                raise ValueError("alpha must be non-negative")
            class_alpha = torch.tensor(float(alpha), dtype=torch.float32)
        else:
            class_alpha = torch.as_tensor(alpha, dtype=torch.float32)
            if class_alpha.ndim != 1 or class_alpha.numel() != num_classes:
                raise ValueError("class-wise alpha must contain num_classes values")
            if torch.any(class_alpha < 0):
                raise ValueError("alpha values must be non-negative")

        self.num_classes = int(num_classes)
        self.gamma = float(gamma)
        self.smooth = float(smooth)
        self.focal_weight = float(focal_weight)
        self.dice_weight = float(dice_weight)
        self.include_background = bool(include_background)
        self.register_buffer("class_alpha", class_alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if not logits.is_floating_point():
            raise TypeError("logits must be a floating-point tensor")
        if logits.ndim < 2 or logits.shape[-1] != self.num_classes:
            raise ValueError(f"logits must have class-last shape (..., {self.num_classes})")
        expected_target_shape = logits.shape[:-1]
        if targets.numel() != math_product(expected_target_shape):
            raise ValueError("targets must contain one class index for every logit vector")

        logits_flat = logits.reshape(-1, self.num_classes)
        targets_flat = targets.to(device=logits.device).reshape(-1)
        _validate_class_targets(targets_flat, self.num_classes)
        targets_flat = targets_flat.long()

        log_probabilities = F.log_softmax(logits_flat, dim=-1)
        log_probability_true_class = log_probabilities.gather(1, targets_flat.unsqueeze(1)).squeeze(
            1
        )
        probability_true_class = log_probability_true_class.exp()

        focal_terms = -((1.0 - probability_true_class).pow(self.gamma) * log_probability_true_class)
        if self.class_alpha is not None:
            alpha = self.class_alpha.to(device=logits.device, dtype=logits.dtype)
            if alpha.ndim == 0:
                focal_terms = focal_terms * alpha
            else:
                focal_terms = focal_terms * alpha[targets_flat]
        focal = focal_terms.mean()

        probabilities = log_probabilities.exp()
        targets_one_hot = F.one_hot(targets_flat, num_classes=self.num_classes).to(
            dtype=logits.dtype
        )
        intersection = torch.sum(probabilities * targets_one_hot, dim=0)
        denominator = torch.sum(probabilities, dim=0) + torch.sum(targets_one_hot, dim=0)
        dice_per_class = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        if not self.include_background:
            dice_per_class = dice_per_class[1:]
        dice_loss = 1.0 - dice_per_class.mean()

        return self.focal_weight * focal + self.dice_weight * dice_loss


def math_product(shape: torch.Size | tuple[int, ...]) -> int:
    """Return a product without depending on a recent ``math.prod`` typing stub."""

    product = 1
    for value in shape:
        product *= int(value)
    return product


# Backward-compatible name used by the original notebooks.  The explicit name
# above is preferred because it makes the expected output contract unmistakable.
FocalDiceLoss = MulticlassFocalDiceLoss

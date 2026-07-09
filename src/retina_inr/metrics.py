"""Pure, per-image segmentation metrics with no cross-image state."""

from __future__ import annotations

import numpy as np
import torch

ArrayLike = np.ndarray | torch.Tensor


def _safe_ratio(numerator: int, denominator: int, *, empty_value: float = 1.0) -> float:
    return float(numerator / denominator) if denominator else float(empty_value)


def _target_as_integer_tensor(target: ArrayLike) -> torch.Tensor:
    tensor = torch.as_tensor(target)
    if tensor.numel() == 0:
        raise ValueError("target must not be empty")
    if tensor.is_floating_point():
        if not torch.isfinite(tensor).all() or not torch.equal(tensor, tensor.round()):
            raise ValueError("target must contain finite integer labels")
    return tensor.long()


def binary_segmentation_metrics(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    from_logits: bool = True,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute binary metrics for one complete image.

    ``prediction`` can be a flattened array, an image, or an image with a
    singleton logit channel.  Counts and all derived values are returned as
    Python floats so the result is directly JSON serializable.  A ratio with an
    empty denominator is defined as ``1.0``; benchmark reports should state that
    convention and retain per-image counts.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    target_tensor = _target_as_integer_tensor(target).reshape(-1)
    if not torch.all((target_tensor == 0) | (target_tensor == 1)):
        raise ValueError("binary target must contain only 0 and 1")

    prediction_tensor = torch.as_tensor(prediction)
    if prediction_tensor.numel() != target_tensor.numel():
        raise ValueError("prediction and target must contain the same number of pixels")
    prediction_tensor = prediction_tensor.reshape(-1)
    if not torch.isfinite(prediction_tensor).all():
        raise ValueError("prediction contains non-finite values")
    probabilities = (
        torch.sigmoid(prediction_tensor.float()) if from_logits else prediction_tensor.float()
    )
    if not from_logits and (torch.any(probabilities < 0) or torch.any(probabilities > 1)):
        raise ValueError("probabilities must be in [0, 1]")

    predicted_positive = probabilities >= threshold
    actual_positive = target_tensor.bool()
    tp = int(torch.sum(predicted_positive & actual_positive))
    tn = int(torch.sum(~predicted_positive & ~actual_positive))
    fp = int(torch.sum(predicted_positive & ~actual_positive))
    fn = int(torch.sum(~predicted_positive & actual_positive))

    return {
        "accuracy": _safe_ratio(tp + tn, tp + tn + fp + fn),
        "dice": _safe_ratio(2 * tp, 2 * tp + fp + fn),
        "iou": _safe_ratio(tp, tp + fp + fn),
        "precision": _safe_ratio(tp, tp + fp),
        "sensitivity": _safe_ratio(tp, tp + fn),
        "specificity": _safe_ratio(tn, tn + fp),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def multiclass_segmentation_metrics(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    num_classes: int,
    from_logits: bool = True,
    include_background: bool = True,
) -> dict[str, float]:
    """Compute class-wise and macro metrics for one complete image.

    With ``from_logits=True``, prediction must be class-last ``(..., K)``.
    With ``from_logits=False``, prediction may be class probabilities with that
    shape or integer labels with the same shape as ``target``.  When a class is
    absent from both prediction and target, its empty-denominator ratios are
    defined as ``1.0``; callers should document or replace that convention when
    aggregating a scientific benchmark.
    """

    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    target_tensor = _target_as_integer_tensor(target).reshape(-1)
    if torch.any(target_tensor < 0) or torch.any(target_tensor >= num_classes):
        raise ValueError(f"target labels must be in [0, {num_classes - 1}]")

    prediction_tensor = torch.as_tensor(prediction)
    if from_logits:
        if prediction_tensor.ndim < 2 or prediction_tensor.shape[-1] != num_classes:
            raise ValueError(f"logits must have class-last shape (..., {num_classes})")
        if prediction_tensor.numel() != target_tensor.numel() * num_classes:
            raise ValueError("prediction and target pixel counts do not match")
        if not torch.isfinite(prediction_tensor).all():
            raise ValueError("prediction contains non-finite values")
        predicted_labels = prediction_tensor.reshape(-1, num_classes).argmax(dim=1)
    elif prediction_tensor.numel() == target_tensor.numel():
        predicted_labels = _target_as_integer_tensor(prediction_tensor).reshape(-1)
    else:
        if prediction_tensor.shape[-1] != num_classes:
            raise ValueError("probabilities must use a class-last dimension")
        if prediction_tensor.numel() != target_tensor.numel() * num_classes:
            raise ValueError("prediction and target pixel counts do not match")
        if not torch.isfinite(prediction_tensor).all():
            raise ValueError("prediction contains non-finite values")
        if torch.any(prediction_tensor < 0) or torch.any(prediction_tensor > 1):
            raise ValueError("probabilities must be in [0, 1]")
        predicted_labels = prediction_tensor.reshape(-1, num_classes).argmax(dim=1)

    if torch.any(predicted_labels < 0) or torch.any(predicted_labels >= num_classes):
        raise ValueError(f"predicted labels must be in [0, {num_classes - 1}]")

    metrics: dict[str, float] = {
        "accuracy": float(torch.mean((predicted_labels == target_tensor).float()))
    }
    dice_values: list[float] = []
    iou_values: list[float] = []
    precision_values: list[float] = []
    sensitivity_values: list[float] = []
    for class_index in range(num_classes):
        predicted_class = predicted_labels == class_index
        actual_class = target_tensor == class_index
        tp = int(torch.sum(predicted_class & actual_class))
        fp = int(torch.sum(predicted_class & ~actual_class))
        fn = int(torch.sum(~predicted_class & actual_class))
        dice = _safe_ratio(2 * tp, 2 * tp + fp + fn)
        iou = _safe_ratio(tp, tp + fp + fn)
        precision = _safe_ratio(tp, tp + fp)
        sensitivity = _safe_ratio(tp, tp + fn)
        metrics.update(
            {
                f"class_{class_index}_dice": dice,
                f"class_{class_index}_iou": iou,
                f"class_{class_index}_precision": precision,
                f"class_{class_index}_sensitivity": sensitivity,
            }
        )
        if include_background or class_index != 0:
            dice_values.append(dice)
            iou_values.append(iou)
            precision_values.append(precision)
            sensitivity_values.append(sensitivity)

    metrics["macro_dice"] = float(sum(dice_values) / len(dice_values))
    metrics["macro_iou"] = float(sum(iou_values) / len(iou_values))
    metrics["macro_precision"] = float(sum(precision_values) / len(precision_values))
    metrics["macro_sensitivity"] = float(sum(sensitivity_values) / len(sensitivity_values))
    return metrics

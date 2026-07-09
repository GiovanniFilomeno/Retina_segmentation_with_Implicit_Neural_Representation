"""Memory-bounded full-image inference and mask-aware evaluation utilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .datasets import (
    Patcher,
    encode_binary_mask,
    encode_multiclass_mask,
    image_to_features,
    normalize_grayscale_image,
    validate_class_labels,
)
from .metrics import binary_segmentation_metrics, multiclass_segmentation_metrics


@dataclass(frozen=True)
class EvaluationResult:
    """Full-image prediction, probabilities, logits, metrics, and optional loss."""

    prediction: np.ndarray
    probabilities: np.ndarray
    logits: np.ndarray
    metrics: dict[str, float]
    loss: float | None


def _parameter_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def chunked_forward(
    model: nn.Module,
    features: torch.Tensor,
    *,
    chunk_size: int = 65_536,
) -> torch.Tensor:
    """Evaluate point features in bounded chunks without changing their order."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if features.ndim != 2 or features.shape[1] != 3:
        raise ValueError("features must have shape (N, 3)")
    if features.shape[0] == 0:
        raise ValueError("features must not be empty")

    outputs: list[torch.Tensor] = []
    for start in range(0, features.shape[0], chunk_size):
        output = model(features[start : start + chunk_size])
        if output.ndim != 2 or output.shape[0] != min(chunk_size, features.shape[0] - start):
            raise ValueError("model must return one class-last logit vector per pixel")
        outputs.append(output)
    return torch.cat(outputs, dim=0)


def predict_image_logits(
    model: nn.Module,
    image: np.ndarray | torch.Tensor,
    *,
    patch_size: int | tuple[int, int] = 256,
    chunk_size: int = 65_536,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Predict one full image and return CPU logits with shape ``(C, H, W)``.

    Images are normalized once by :func:`normalize_grayscale_image`.  Each
    non-overlapping patch receives the same patch-local coordinate grid used by
    the training dataset.  Chunking limits activation memory and preserves the
    exact row-major order needed for reconstruction.
    """

    normalized = normalize_grayscale_image(image).unsqueeze(0)
    patcher = Patcher(patch_size, padding_mode="reflect")
    image_patches, metadata = patcher.patch(normalized)
    original_device = _parameter_device(model)
    inference_device = torch.device(device) if device is not None else original_device
    model.to(inference_device)
    was_training = model.training
    predicted_patches: list[torch.Tensor] = []

    try:
        model.eval()
        with torch.inference_mode():
            output_channels: int | None = None
            for image_patch in image_patches:
                features = image_to_features(image_patch, device=inference_device)
                logits = chunked_forward(model, features, chunk_size=chunk_size)
                if output_channels is None:
                    output_channels = logits.shape[-1]
                    if output_channels < 1:
                        raise ValueError("model returned no output channels")
                elif logits.shape[-1] != output_channels:
                    raise ValueError("model output channel count changed between patches")
                patch_h, patch_w = image_patch.shape[-2:]
                predicted_patches.append(
                    logits.reshape(patch_h, patch_w, output_channels)
                    .permute(2, 0, 1)
                    .detach()
                    .cpu()
                )
    finally:
        model.train(was_training)
        if inference_device != original_device:
            model.to(original_device)

    return patcher.unpatch(torch.stack(predicted_patches, dim=0), metadata)


def evaluate_binary_model(
    model: nn.Module,
    image: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    *,
    patch_size: int | tuple[int, int] = 256,
    chunk_size: int = 65_536,
    threshold: float = 0.5,
    criterion: nn.Module | None = None,
    device: torch.device | str | None = None,
) -> EvaluationResult:
    """Evaluate a one-logit model against the supplied binary mask."""

    logits = predict_image_logits(
        model,
        image,
        patch_size=patch_size,
        chunk_size=chunk_size,
        device=device,
    )
    if logits.shape[0] != 1:
        raise ValueError(f"binary evaluation requires one logit channel, got {logits.shape[0]}")
    target = encode_binary_mask(mask)
    if tuple(target.shape) != tuple(logits.shape[-2:]):
        raise ValueError(
            f"image/mask shape mismatch: prediction {tuple(logits.shape[-2:])}, "
            f"mask {tuple(target.shape)}"
        )

    logit_image = logits.squeeze(0)
    metrics = binary_segmentation_metrics(
        logit_image, target, from_logits=True, threshold=threshold
    )
    probabilities = torch.sigmoid(logit_image)
    prediction = probabilities >= threshold
    loss = None
    if criterion is not None:
        loss_device = _parameter_device(model)
        with torch.inference_mode():
            loss = float(
                criterion(logit_image.to(loss_device), target.to(loss_device)).detach().cpu()
            )

    return EvaluationResult(
        prediction=prediction.to(torch.uint8).numpy(),
        probabilities=probabilities.numpy(),
        logits=logit_image.numpy(),
        metrics=metrics,
        loss=loss,
    )


def evaluate_multiclass_model(
    model: nn.Module,
    image: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    *,
    num_classes: int | None = None,
    value_to_class: Mapping[int, int] | None = None,
    patch_size: int | tuple[int, int] = 256,
    chunk_size: int = 65_536,
    include_background: bool = True,
    criterion: nn.Module | None = None,
    device: torch.device | str | None = None,
) -> EvaluationResult:
    """Evaluate K-logit predictions against the supplied full-image class mask."""

    logits = predict_image_logits(
        model,
        image,
        patch_size=patch_size,
        chunk_size=chunk_size,
        device=device,
    )
    inferred_classes = logits.shape[0]
    if num_classes is None:
        num_classes = inferred_classes
    if num_classes < 2 or inferred_classes != num_classes:
        raise ValueError(f"expected {num_classes} output channels, got {inferred_classes}")

    if value_to_class is not None:
        target = encode_multiclass_mask(mask, value_to_class)
    else:
        target = torch.as_tensor(mask)
        if target.ndim == 3 and target.shape[0] == 1:
            target = target.squeeze(0)
        elif target.ndim == 3 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        if target.ndim != 2:
            raise ValueError("mask must be a 2-D class-index image")
        target = validate_class_labels(target, num_classes)
    if tuple(target.shape) != tuple(logits.shape[-2:]):
        raise ValueError(
            f"image/mask shape mismatch: prediction {tuple(logits.shape[-2:])}, "
            f"mask {tuple(target.shape)}"
        )

    class_last_logits = logits.permute(1, 2, 0).contiguous()
    metrics = multiclass_segmentation_metrics(
        class_last_logits,
        target,
        num_classes=num_classes,
        from_logits=True,
        include_background=include_background,
    )
    probabilities = torch.softmax(logits, dim=0)
    prediction = torch.argmax(logits, dim=0)
    loss = None
    if criterion is not None:
        loss_device = _parameter_device(model)
        with torch.inference_mode():
            loss = float(
                criterion(class_last_logits.to(loss_device), target.to(loss_device)).detach().cpu()
            )

    return EvaluationResult(
        prediction=prediction.to(torch.uint8).numpy(),
        probabilities=probabilities.numpy(),
        logits=logits.numpy(),
        metrics=metrics,
        loss=loss,
    )


def colorize_segmentation_mask(
    mask: np.ndarray | torch.Tensor,
    colors: Mapping[int, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Convert a class-index mask to an RGB uint8 image without plotting state."""

    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        raise ValueError("mask must be 2-D")
    palette = colors or {0: (0, 0, 0), 1: (0, 90, 255), 2: (255, 70, 40)}
    unknown = set(int(value) for value in np.unique(mask_array)) - set(palette)
    if unknown:
        raise ValueError(f"colors are missing class indices {sorted(unknown)}")
    rgb = np.zeros((*mask_array.shape, 3), dtype=np.uint8)
    for class_index, color in palette.items():
        rgb[mask_array == class_index] = color
    return rgb

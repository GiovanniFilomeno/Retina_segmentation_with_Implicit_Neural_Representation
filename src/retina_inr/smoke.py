"""Fast, data-free CPU smoke checks for the public package API."""

from __future__ import annotations

import json

import numpy as np
import torch

from .datasets import Patcher, image_to_features
from .losses import BinaryFocalDiceLoss, MulticlassFocalDiceLoss
from .models import INRSegmentationModel


def _finite_gradients(model: torch.nn.Module) -> bool:
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    return bool(gradients) and all(torch.isfinite(gradient).all() for gradient in gradients)


def run_smoke_checks() -> dict[str, bool]:
    """Run deterministic preprocessing, geometry, forward, loss, and backward checks.

    The function returns only if every assertion succeeds.  It downloads no
    data, uses no checkpoints, and is intentionally small enough for CPU CI.
    """

    torch.manual_seed(7)
    image = np.arange(35, dtype=np.uint8).reshape(5, 7)
    features = image_to_features(image)
    if features.shape != (35, 3) or not torch.isfinite(features).all():
        raise AssertionError("preprocessing smoke check failed")

    source = torch.arange(35, dtype=torch.float32).reshape(1, 5, 7)
    patcher = Patcher((4, 4))
    patches, metadata = patcher.patch(source)
    reconstructed = patcher.unpatch(patches, metadata)
    patch_roundtrip = bool(torch.equal(source, reconstructed))
    if not patch_roundtrip:
        raise AssertionError("patch/unpatch roundtrip failed")

    binary_model = INRSegmentationModel(task="binary", hidden_dim=16, num_layers=2, num_freqs=2)
    binary_logits = binary_model(features)
    binary_forward = bool(binary_logits.shape == (35, 1) and torch.isfinite(binary_logits).all())
    if not binary_forward:
        raise AssertionError("binary forward smoke check failed")
    binary_targets = (torch.arange(35) % 5 == 0).float()
    binary_loss = BinaryFocalDiceLoss()(binary_logits, binary_targets)
    binary_loss.backward()

    multiclass_model = INRSegmentationModel(
        task="multiclass",
        num_classes=3,
        hidden_dim=16,
        num_layers=2,
        num_freqs=2,
    )
    multiclass_logits = multiclass_model(features)
    multiclass_forward = bool(
        multiclass_logits.shape == (35, 3) and torch.isfinite(multiclass_logits).all()
    )
    if not multiclass_forward:
        raise AssertionError("multiclass forward smoke check failed")
    multiclass_targets = torch.arange(35) % 3
    multiclass_loss = MulticlassFocalDiceLoss(3)(multiclass_logits, multiclass_targets)
    multiclass_loss.backward()

    losses_finite = bool(
        torch.isfinite(binary_loss)
        and torch.isfinite(multiclass_loss)
        and _finite_gradients(binary_model)
        and _finite_gradients(multiclass_model)
    )
    if not losses_finite:
        raise AssertionError("loss/backward smoke check failed")

    return {
        "patch_roundtrip": patch_roundtrip,
        "binary_forward": binary_forward,
        "multiclass_forward": multiclass_forward,
        "losses_finite": losses_finite,
        "preprocessing": True,
        "backward": True,
    }


def main() -> None:
    """CLI entry point for ``python -m retina_inr.smoke``."""

    print(json.dumps(run_smoke_checks(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

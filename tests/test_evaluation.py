import numpy as np
import pytest
import torch
from torch import nn

from retina_inr import (
    RAVIR_VALUE_TO_CLASS,
    BinaryFocalDiceLoss,
    MulticlassFocalDiceLoss,
    evaluate_binary_model,
    evaluate_multiclass_model,
)


class BinaryIntensityModel(nn.Module):
    """Classify normalized intensities above 0.5 as foreground."""

    def forward(self, features):
        return (features[:, 2:3] - 0.5) * 40.0


class MulticlassIntensityModel(nn.Module):
    """Map the three RAVIR grayscale values to three class logits."""

    def forward(self, features):
        centers = features.new_tensor([0.0, 128.0 / 255.0, 1.0])
        return -torch.abs(features[:, 2:3] - centers) * 40.0


def test_binary_evaluation_uses_mask_and_reconstructs_odd_image():
    image = np.array(
        [
            [0, 255, 0, 255, 0],
            [255, 0, 255, 0, 255],
            [0, 0, 255, 255, 0],
        ],
        dtype=np.uint8,
    )
    mask = (image == 255).astype(np.uint8)
    model = BinaryIntensityModel().train()

    result = evaluate_binary_model(
        model,
        image,
        mask,
        patch_size=(2, 3),
        chunk_size=4,
        criterion=BinaryFocalDiceLoss(),
    )

    assert model.training is True
    assert result.prediction.shape == image.shape
    assert np.array_equal(result.prediction, mask)
    assert result.metrics["dice"] == pytest.approx(1.0)
    assert result.metrics["iou"] == pytest.approx(1.0)
    assert result.loss is not None
    assert result.loss < 1e-6


def test_multiclass_evaluation_aligns_patches_with_full_mask():
    image = np.array(
        [
            [0, 128, 255, 0, 128],
            [255, 0, 128, 255, 0],
            [128, 255, 0, 128, 255],
        ],
        dtype=np.uint8,
    )
    expected = np.vectorize(RAVIR_VALUE_TO_CLASS.__getitem__)(image).astype(np.uint8)
    model = MulticlassIntensityModel().train()

    result = evaluate_multiclass_model(
        model,
        image,
        image,
        num_classes=3,
        value_to_class=RAVIR_VALUE_TO_CLASS,
        patch_size=(2, 3),
        chunk_size=4,
        criterion=MulticlassFocalDiceLoss(3),
    )

    assert model.training is True
    assert result.prediction.shape == image.shape
    assert np.array_equal(result.prediction, expected)
    assert result.metrics["macro_dice"] == pytest.approx(1.0)
    assert result.metrics["macro_iou"] == pytest.approx(1.0)
    assert result.loss is not None
    assert result.loss < 1e-3

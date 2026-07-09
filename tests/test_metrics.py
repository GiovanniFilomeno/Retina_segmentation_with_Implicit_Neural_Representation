import pytest
import torch

from retina_inr import binary_segmentation_metrics, multiclass_segmentation_metrics


def test_binary_metrics_are_perfect_for_perfect_logits():
    targets = torch.tensor([0, 1, 1, 0, 1, 0])
    logits = torch.where(targets.bool(), 20.0, -20.0)

    metrics = binary_segmentation_metrics(logits, targets)

    for name in ("accuracy", "dice", "iou", "precision", "sensitivity", "specificity"):
        assert metrics[name] == pytest.approx(1.0)
    assert metrics["tp"] == pytest.approx(3.0)
    assert metrics["tn"] == pytest.approx(3.0)
    assert metrics["fp"] == pytest.approx(0.0)
    assert metrics["fn"] == pytest.approx(0.0)


def test_multiclass_metrics_are_perfect_for_perfect_logits():
    targets = torch.tensor([0, 1, 2, 0, 1, 2])
    logits = torch.full((targets.numel(), 3), -20.0)
    logits[torch.arange(targets.numel()), targets] = 20.0

    metrics = multiclass_segmentation_metrics(logits, targets, num_classes=3)

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["macro_dice"] == pytest.approx(1.0)
    assert metrics["macro_iou"] == pytest.approx(1.0)
    for class_index in range(3):
        for metric_name in ("dice", "iou", "precision", "sensitivity"):
            assert metrics[f"class_{class_index}_{metric_name}"] == pytest.approx(1.0)

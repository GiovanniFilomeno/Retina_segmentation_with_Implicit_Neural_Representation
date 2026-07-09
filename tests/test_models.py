import pytest
import torch

from retina_inr import INRSegmentationModel


@pytest.mark.parametrize(
    ("task", "num_classes", "expected_channels"),
    [
        ("binary", None, 1),
        ("multiclass", 3, 3),
    ],
)
def test_model_heads_return_per_pixel_logits(task, num_classes, expected_channels):
    torch.manual_seed(0)
    model = INRSegmentationModel(
        task=task,
        num_classes=num_classes,
        hidden_dim=16,
        num_layers=3,
        num_freqs=2,
        dropout=0.0,
    ).eval()
    features = torch.rand(11, 3)

    logits = model(features)

    assert logits.shape == (11, expected_channels)
    assert logits.requires_grad
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize(
    ("task", "num_classes"),
    [
        ("not-a-task", None),
        ("binary", 2),
        ("multiclass", 1),
    ],
)
def test_model_rejects_invalid_task_configuration(task, num_classes):
    with pytest.raises(ValueError, match=r"task|binary|num_classes"):
        INRSegmentationModel(task=task, num_classes=num_classes)

import torch

from retina_inr import BinaryFocalDiceLoss, INRSegmentationModel, image_to_features
from retina_inr.smoke import run_smoke_checks


def test_bundled_smoke_checks_pass():
    checks = run_smoke_checks()

    assert {
        "patch_roundtrip",
        "binary_forward",
        "multiclass_forward",
        "losses_finite",
    } <= checks.keys()
    assert all(checks.values())


def test_synthetic_binary_training_step_is_finite():
    torch.manual_seed(7)
    image = torch.linspace(0.0, 1.0, steps=30).reshape(5, 6)
    features = image_to_features(image)
    targets = (image.reshape(-1, 1) > 0.5).float()
    model = INRSegmentationModel(
        task="binary",
        hidden_dim=16,
        num_layers=3,
        num_freqs=2,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    optimizer.zero_grad(set_to_none=True)
    loss = BinaryFocalDiceLoss()(model(features), targets)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())

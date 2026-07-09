import torch

from retina_inr import BinaryFocalDiceLoss, MulticlassFocalDiceLoss


def test_binary_loss_is_finite_for_extreme_logits_and_backpropagates():
    logits = torch.tensor([[-1_000.0], [1_000.0], [0.0], [25.0]], requires_grad=True)
    targets = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

    loss = BinaryFocalDiceLoss()(logits, targets)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_multiclass_loss_is_finite_for_extreme_logits_and_backpropagates():
    logits = torch.tensor(
        [
            [1_000.0, -1_000.0, 0.0],
            [-1_000.0, 1_000.0, 0.0],
            [0.0, -1_000.0, 1_000.0],
            [25.0, -25.0, 0.0],
        ],
        requires_grad=True,
    )
    targets = torch.tensor([0, 1, 2, 2])

    loss = MulticlassFocalDiceLoss(num_classes=3)(logits, targets)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_better_binary_predictions_have_lower_loss():
    targets = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    criterion = BinaryFocalDiceLoss()

    good = criterion(torch.tensor([[-8.0], [8.0], [8.0], [-8.0]]), targets)
    bad = criterion(torch.tensor([[8.0], [-8.0], [-8.0], [8.0]]), targets)

    assert good < bad


def test_better_multiclass_predictions_have_lower_loss():
    targets = torch.tensor([0, 1, 2])
    criterion = MulticlassFocalDiceLoss(num_classes=3)
    good = torch.tensor([[8.0, -8.0, -8.0], [-8.0, 8.0, -8.0], [-8.0, -8.0, 8.0]])
    bad = -good

    assert criterion(good, targets) < criterion(bad, targets)

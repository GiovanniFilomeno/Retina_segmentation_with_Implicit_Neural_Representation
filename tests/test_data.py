from pathlib import Path

import numpy as np
import pytest
import torch

from retina_inr import (
    Patcher,
    build_training_augmentation,
    encode_binary_mask,
    encode_multiclass_mask,
    image_to_features,
    make_coordinate_grid,
    pair_image_mask_files,
)


def test_patch_roundtrip_preserves_odd_spatial_shape():
    image = torch.arange(2 * 5 * 7, dtype=torch.float32).reshape(2, 5, 7)
    patcher = Patcher((4, 3))

    patches, metadata = patcher.patch(image)
    reconstructed = patcher.unpatch(patches, metadata)

    assert reconstructed.shape == image.shape
    torch.testing.assert_close(reconstructed, image)


def test_coordinate_grid_is_row_major_xy_on_minus_one_to_one():
    grid = make_coordinate_grid(2, 3)
    expected = torch.tensor(
        [
            [[-1.0, -1.0], [0.0, -1.0], [1.0, -1.0]],
            [[-1.0, 1.0], [0.0, 1.0], [1.0, 1.0]],
        ]
    )

    torch.testing.assert_close(grid, expected)


def test_singleton_coordinate_dimension_is_centered():
    grid = make_coordinate_grid(1, 3)

    torch.testing.assert_close(grid[..., 1], torch.zeros(1, 3))


def test_image_features_align_coordinates_and_intensities():
    image = torch.tensor([[0.0, 0.25, 0.5], [0.75, 1.0, 0.125]])

    features = image_to_features(image)

    assert features.shape == (6, 3)
    torch.testing.assert_close(features[:, :2], make_coordinate_grid(2, 3).reshape(-1, 2))
    torch.testing.assert_close(features[:, 2], image.reshape(-1))


def test_pairing_rejects_missing_mask(tmp_path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()
    (images_dir / "case_01.png").touch()
    (images_dir / "case_02.png").touch()
    (masks_dir / "case_01.png").touch()

    with pytest.raises(ValueError, match="case_02"):
        pair_image_mask_files(images_dir, masks_dir)


def test_pairing_matches_by_stem_instead_of_sort_position(tmp_path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()
    for name in ("case_b.png", "case_a.png"):
        (images_dir / name).touch()
    for name in ("case_a.tif", "case_b.tif"):
        (masks_dir / name).touch()

    pairs = pair_image_mask_files(images_dir, masks_dir)

    assert [Path(pair.image_path).stem for pair in pairs] == ["case_a", "case_b"]
    assert [Path(pair.mask_path).stem for pair in pairs] == ["case_a", "case_b"]


def test_mask_encoders_preserve_declared_label_sets():
    binary = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    multiclass = np.array([[0, 128, 255]], dtype=np.uint8)

    encoded_binary = encode_binary_mask(binary)
    encoded_multiclass = encode_multiclass_mask(multiclass)

    assert np.array_equal(np.asarray(encoded_binary), np.array([[0, 1], [1, 0]]))
    assert np.array_equal(np.asarray(encoded_multiclass), np.array([[0, 1, 2]]))


def test_multiclass_encoder_rejects_corrupted_mask_values():
    corrupted = np.array([[0, 127, 255]], dtype=np.uint8)

    with pytest.raises(ValueError, match="127"):
        encode_multiclass_mask(corrupted)


def test_training_augmentation_preserves_categorical_mask_values(monkeypatch):
    monkeypatch.setenv("NO_ALBUMENTATIONS_UPDATE", "1")
    image = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    mask = np.where((np.indices((64, 64)).sum(axis=0) % 2) == 0, 0, 255).astype(np.uint8)
    augmentation = build_training_augmentation()

    for _ in range(12):
        transformed = augmentation(image=image, mask=mask)

        assert transformed["image"].shape == image.shape
        assert transformed["mask"].shape == mask.shape
        assert set(np.unique(transformed["mask"])) <= {0, 255}

"""Validated data and patch utilities for pointwise retinal segmentation.

Coordinates generated here are patch-local: every patch spans ``[-1, 1]`` in
both axes.  This matches training and inference, but it also means that the
model has no explicit global-image position and no neighbourhood context.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

VALID_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})
DEFAULT_MASK_SUFFIXES = (
    "_mask",
    "-mask",
    "_label",
    "-label",
    "_manual",
    "_segmentation",
    "_seg",
)
RAVIR_VALUE_TO_CLASS = {0: 0, 128: 1, 255: 2}


@dataclass(frozen=True)
class ImageMaskPair:
    """One image/mask pair matched by a validated canonical key."""

    key: str
    image_path: Path
    mask_path: Path


def _canonical_stem(path: Path, suffixes: Sequence[str]) -> str:
    stem = path.stem.casefold()
    for suffix in sorted(suffixes, key=len, reverse=True):
        suffix_folded = suffix.casefold()
        if stem.endswith(suffix_folded):
            stem = stem[: -len(suffix_folded)]
            break
    return stem


def _indexed_files(
    directory: str | Path,
    *,
    suffixes: Sequence[str],
    key_fn: Callable[[Path], str] | None,
) -> dict[str, Path]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"not a directory: {root}")
    paths = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.casefold() in VALID_IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.casefold(),
    )
    if not paths:
        raise ValueError(f"no supported image files found in {root}")

    indexed: dict[str, Path] = {}
    for path in paths:
        key = key_fn(path) if key_fn is not None else _canonical_stem(path, suffixes)
        key = str(key).strip().casefold()
        if not key:
            raise ValueError(f"empty pairing key generated for {path}")
        if key in indexed:
            raise ValueError(
                f"duplicate pairing key {key!r}: {indexed[key].name!r} and {path.name!r}"
            )
        indexed[key] = path
    return indexed


def pair_image_mask_files(
    images_dir: str | Path,
    masks_dir: str | Path,
    *,
    mask_suffixes: Sequence[str] = DEFAULT_MASK_SUFFIXES,
    key_fn: Callable[[Path], str] | None = None,
) -> list[ImageMaskPair]:
    """Pair images and masks by canonical stem and reject any mismatch.

    Independent lexicographic sorting is unsafe because one missing file shifts
    every subsequent pair.  This function instead builds unique stem indexes,
    strips common mask suffixes, and fails with the exact missing keys.
    """

    images = _indexed_files(images_dir, suffixes=mask_suffixes, key_fn=key_fn)
    masks = _indexed_files(masks_dir, suffixes=mask_suffixes, key_fn=key_fn)
    image_only = sorted(images.keys() - masks.keys())
    mask_only = sorted(masks.keys() - images.keys())
    if image_only or mask_only:
        details: list[str] = []
        if image_only:
            details.append(f"images without masks: {image_only}")
        if mask_only:
            details.append(f"masks without images: {mask_only}")
        raise ValueError("image/mask pairing failed; " + "; ".join(details))

    return [
        ImageMaskPair(key=key, image_path=images[key], mask_path=masks[key])
        for key in sorted(images)
    ]


def normalize_grayscale_image(
    image: np.ndarray | torch.Tensor,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return one grayscale image as ``(H, W)`` floating point in ``[0, 1]``.

    Integer arrays are scaled by their dtype maximum (for example 255 for
    ``uint8``).  Floating-point inputs must already be normalized; values such
    as floating-point ``[0, 255]`` are rejected so training and inference cannot
    silently use different intensity ranges.
    """

    tensor = torch.as_tensor(image, device=device)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    elif tensor.ndim == 3 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    if tensor.ndim != 2:
        raise ValueError("image must be single-channel with shape (H, W) or (1, H, W)")
    if tensor.numel() == 0:
        raise ValueError("image must not be empty")

    if tensor.is_floating_point():
        normalized = tensor.to(dtype=dtype)
        if not torch.isfinite(normalized).all():
            raise ValueError("image contains non-finite values")
        lower = float(normalized.min())
        upper = float(normalized.max())
        if lower < -1e-6 or upper > 1.0 + 1e-6:
            raise ValueError("floating-point images must already be normalized to [0, 1]")
        return normalized.clamp(0.0, 1.0)

    if tensor.dtype == torch.bool:
        return tensor.to(dtype=dtype)
    info = torch.iinfo(tensor.dtype)
    if info.min < 0 and torch.any(tensor < 0):
        raise ValueError("integer images must be non-negative")
    return tensor.to(dtype=dtype) / float(info.max)


def make_coordinate_grid(
    height: int,
    width: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a row-major ``(H, W, 2)`` patch-local grid with columns ``[x, y]``.

    Each non-singleton axis spans ``[-1, 1]`` including both endpoints.  A
    singleton axis is assigned coordinate zero.
    """

    if height < 1 or width < 1:
        raise ValueError("height and width must be positive")
    if not dtype.is_floating_point:
        raise TypeError("coordinate dtype must be floating point")

    y = (
        torch.zeros(1, device=device, dtype=dtype)
        if height == 1
        else torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    )
    x = (
        torch.zeros(1, device=device, dtype=dtype)
        if width == 1
        else torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    )
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1)


def image_to_features(
    image: np.ndarray | torch.Tensor,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert a grayscale image into row-major ``[x, y, intensity]`` rows."""

    normalized = normalize_grayscale_image(image, device=device, dtype=dtype)
    height, width = normalized.shape
    coordinates = make_coordinate_grid(
        height, width, device=normalized.device, dtype=normalized.dtype
    )
    return torch.cat((coordinates.reshape(-1, 2), normalized.reshape(-1, 1)), dim=1)


def _integral_mask(mask: np.ndarray | torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(mask)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    elif tensor.ndim == 3 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    if tensor.ndim != 2:
        raise ValueError("mask must have shape (H, W) or one singleton channel")
    if tensor.numel() == 0:
        raise ValueError("mask must not be empty")
    if tensor.is_floating_point():
        if not torch.isfinite(tensor).all() or not torch.equal(tensor, tensor.round()):
            raise ValueError("mask values must be finite integers")
    return tensor.long()


def encode_binary_mask(mask: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Strictly encode a ``{0, 1}`` or ``{0, 255}`` mask as integer labels."""

    tensor = _integral_mask(mask)
    values = set(int(value) for value in torch.unique(tensor).tolist())
    if values <= {0, 1}:
        return tensor
    if values <= {0, 255}:
        return (tensor == 255).long()
    raise ValueError(
        f"binary mask contains unsupported values {sorted(values)}; expected {{0, 1}} or {{0, 255}}"
    )


def encode_multiclass_mask(
    mask: np.ndarray | torch.Tensor,
    value_to_class: Mapping[int, int] = RAVIR_VALUE_TO_CLASS,
    *,
    allow_already_encoded: bool = True,
) -> torch.Tensor:
    """Map raw mask values to contiguous class indices with strict validation."""

    if not value_to_class:
        raise ValueError("value_to_class must not be empty")
    mapping = {int(raw): int(label) for raw, label in value_to_class.items()}
    labels = sorted(set(mapping.values()))
    if labels != list(range(len(labels))):
        raise ValueError("mapped class indices must be contiguous and start at zero")

    tensor = _integral_mask(mask)
    values = set(int(value) for value in torch.unique(tensor).tolist())
    if values <= set(mapping):
        encoded = torch.empty_like(tensor)
        for raw_value, class_index in mapping.items():
            encoded[tensor == raw_value] = class_index
        return encoded
    if allow_already_encoded and values <= set(labels):
        return tensor
    unknown = sorted(values - set(mapping))
    raise ValueError(f"mask contains values absent from value_to_class: {unknown}")


def validate_class_labels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Validate and return class labels as ``torch.long``."""

    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    tensor = torch.as_tensor(labels)
    if tensor.is_floating_point() and not torch.equal(tensor, tensor.round()):
        raise ValueError("labels must be integer class indices")
    tensor = tensor.long()
    if tensor.numel() == 0 or torch.any(tensor < 0) or torch.any(tensor >= num_classes):
        raise ValueError(f"labels must be non-empty and in [0, {num_classes - 1}]")
    return tensor


@dataclass(frozen=True)
class PatchMetadata:
    """Geometry required to invert a non-overlapping patch operation."""

    original_shape: tuple[int, int]
    padded_shape: tuple[int, int]
    grid_shape: tuple[int, int]
    patch_size: tuple[int, int]


class Patcher:
    """Split and exactly reassemble channel-first tensors using row-major patches."""

    def __init__(
        self,
        patch_size: int | tuple[int, int],
        *,
        padding_mode: str = "reflect",
    ) -> None:
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if len(patch_size) != 2 or any(value < 1 for value in patch_size):
            raise ValueError("patch_size must contain two positive integers")
        if padding_mode not in {"reflect", "replicate", "constant"}:
            raise ValueError("padding_mode must be 'reflect', 'replicate', or 'constant'")
        self.patch_size = (int(patch_size[0]), int(patch_size[1]))
        self.padding_mode = padding_mode

    def patch(self, tensor: torch.Tensor) -> tuple[torch.Tensor, PatchMetadata]:
        if tensor.ndim != 3:
            raise ValueError("tensor must have shape (C, H, W)")
        channels, height, width = tensor.shape
        if channels < 1 or height < 1 or width < 1:
            raise ValueError("tensor dimensions must be positive")
        patch_h, patch_w = self.patch_size
        pad_h = (-height) % patch_h
        pad_w = (-width) % patch_w

        mode = self.padding_mode
        if mode == "reflect" and (
            (pad_h > 0 and pad_h >= height) or (pad_w > 0 and pad_w >= width)
        ):
            mode = "replicate"
        padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode)
        padded_h, padded_w = height + pad_h, width + pad_w
        grid_h, grid_w = padded_h // patch_h, padded_w // patch_w
        patches = (
            padded.unfold(1, patch_h, patch_h)
            .unfold(2, patch_w, patch_w)
            .permute(1, 2, 0, 3, 4)
            .contiguous()
            .reshape(grid_h * grid_w, channels, patch_h, patch_w)
        )
        metadata = PatchMetadata(
            original_shape=(height, width),
            padded_shape=(padded_h, padded_w),
            grid_shape=(grid_h, grid_w),
            patch_size=self.patch_size,
        )
        return patches, metadata

    def unpatch(self, patches: torch.Tensor, metadata: PatchMetadata) -> torch.Tensor:
        if patches.ndim != 4:
            raise ValueError("patches must have shape (P, C, patch_H, patch_W)")
        if metadata.patch_size != self.patch_size:
            raise ValueError("metadata patch size does not match this Patcher")
        grid_h, grid_w = metadata.grid_shape
        patch_h, patch_w = self.patch_size
        expected = grid_h * grid_w
        if patches.shape[0] != expected or tuple(patches.shape[-2:]) != self.patch_size:
            raise ValueError(
                f"expected {expected} patches of size {self.patch_size}, got {tuple(patches.shape)}"
            )
        channels = patches.shape[1]
        padded = (
            patches.reshape(grid_h, grid_w, channels, patch_h, patch_w)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
            .reshape(channels, grid_h * patch_h, grid_w * patch_w)
        )
        height, width = metadata.original_shape
        return padded[:, :height, :width]


class Augmentation(Protocol):
    """Albumentations-compatible callable protocol."""

    def __call__(self, *, image: np.ndarray, mask: np.ndarray) -> Mapping[str, Any]: ...


def build_training_augmentation() -> Augmentation:
    """Build a conservative augmentation pipeline with masks typed correctly.

    Albumentations already recognizes the keyword ``mask`` as a label target:
    geometric transforms use nearest-neighbour interpolation, while brightness,
    contrast, noise, blur, and CLAHE affect only ``image``.  In particular, the
    mask is never registered as an additional ``image`` target.
    """

    try:
        import albumentations as A
    except ImportError as error:  # pragma: no cover - dependency error is explicit
        raise ImportError("augmentation requires the 'albumentations' package") from error

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT_101, p=0.4),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussianBlur(blur_limit=(3, 3), p=0.1),
            A.CLAHE(clip_limit=2.0, p=0.1),
        ]
    )


def _read_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"OpenCV could not read {path}")
    return image


class RetinaPatchDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Validated image/mask dataset returning one pointwise patch per item."""

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        *,
        task: str,
        patch_size: int | tuple[int, int] = 256,
        num_classes: int | None = None,
        value_to_class: Mapping[int, int] | None = None,
        target_size: tuple[int, int] | None = None,
        augmentation: bool | Augmentation = False,
    ) -> None:
        if task not in {"binary", "multiclass"}:
            raise ValueError("task must be 'binary' or 'multiclass'")
        if task == "multiclass" and (num_classes is None or num_classes < 2):
            raise ValueError("multiclass datasets require num_classes >= 2")
        if target_size is not None and any(value < 1 for value in target_size):
            raise ValueError("target_size must contain positive (height, width)")

        self.task = task
        self.num_classes = 1 if task == "binary" else int(num_classes or 0)
        self.value_to_class = value_to_class
        self.target_size = target_size
        self.pairs = pair_image_mask_files(images_dir, masks_dir)
        self.image_patcher = Patcher(patch_size, padding_mode="reflect")
        self.mask_patcher = Patcher(patch_size, padding_mode="replicate")
        self.augmentation = (
            build_training_augmentation() if augmentation is True else augmentation or None
        )

        self._patch_index: list[tuple[int, int]] = []
        patch_h, patch_w = self.image_patcher.patch_size
        for pair_index, pair in enumerate(self.pairs):
            image = _read_grayscale(pair.image_path)
            mask = _read_grayscale(pair.mask_path)
            if image.shape != mask.shape:
                raise ValueError(
                    f"shape mismatch for {pair.key!r}: image {image.shape}, mask {mask.shape}"
                )
            height, width = target_size or image.shape
            patch_count = math.ceil(height / patch_h) * math.ceil(width / patch_w)
            self._patch_index.extend(
                (pair_index, patch_index) for patch_index in range(patch_count)
            )

    def __len__(self) -> int:
        return len(self._patch_index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair_index, patch_index = self._patch_index[index]
        pair = self.pairs[pair_index]
        image = _read_grayscale(pair.image_path)
        mask = _read_grayscale(pair.mask_path)
        if image.shape != mask.shape:
            raise ValueError(f"shape mismatch for pair {pair.key!r}")

        if self.target_size is not None:
            height, width = self.target_size
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        if self.augmentation is not None:
            transformed = self.augmentation(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]
            if image.shape[:2] != mask.shape[:2]:
                raise ValueError("augmentation returned different image and mask shapes")

        image_tensor = normalize_grayscale_image(image).unsqueeze(0)
        if self.task == "binary":
            label_tensor = encode_binary_mask(mask)
        else:
            if self.value_to_class is None:
                label_tensor = validate_class_labels(_integral_mask(mask), self.num_classes)
            else:
                label_tensor = encode_multiclass_mask(mask, self.value_to_class)
                label_tensor = validate_class_labels(label_tensor, self.num_classes)

        image_patches, image_metadata = self.image_patcher.patch(image_tensor)
        mask_patches, mask_metadata = self.mask_patcher.patch(label_tensor.unsqueeze(0))
        if image_metadata != mask_metadata:
            raise RuntimeError("image and mask patch geometry diverged")
        image_patch = image_patches[patch_index]
        mask_patch = mask_patches[patch_index]
        features = image_to_features(image_patch)
        labels = mask_patch.reshape(-1).long()
        return features, labels


class FIVESDataset(RetinaPatchDataset):
    """Binary FIVES dataset with validated file pairing and patch geometry."""

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        *,
        patch_size: int | tuple[int, int] = 256,
        target_size: tuple[int, int] | None = None,
        augmentation: bool | Augmentation = False,
    ) -> None:
        super().__init__(
            images_dir,
            masks_dir,
            task="binary",
            patch_size=patch_size,
            target_size=target_size,
            augmentation=augmentation,
        )


class RAVIRDataset(RetinaPatchDataset):
    """Three-class RAVIR dataset using the published ``0/128/255`` labels."""

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        *,
        patch_size: int | tuple[int, int] = 256,
        target_size: tuple[int, int] | None = None,
        augmentation: bool | Augmentation = False,
    ) -> None:
        super().__init__(
            images_dir,
            masks_dir,
            task="multiclass",
            num_classes=3,
            value_to_class=RAVIR_VALUE_TO_CLASS,
            patch_size=patch_size,
            target_size=target_size,
            augmentation=augmentation,
        )


def concatenate_variable_length_batch(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate point samples when patch sizes vary between items."""

    if not batch:
        raise ValueError("batch must not be empty")
    features, labels = zip(*batch, strict=True)
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


ravir_collate_fn = concatenate_variable_length_batch

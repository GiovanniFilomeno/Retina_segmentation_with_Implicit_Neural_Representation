"""
Dataset classes for retinal vessel segmentation with INR models.

This module provides PyTorch Dataset implementations for two retinal imaging
benchmarks:

1. **FIVES** (Fundus Image VEssel Segmentation): High-resolution 2048x2048
   color fundus photographs with binary vessel masks. Images are split into
   non-overlapping patches for training.

2. **RAVIR** (Retinal Artery-Vein Segmentation in IR): 768x768 infrared
   angiography images with 3-class masks (background=0, vein=128, artery=255).
   Uses dynamic patching with reflection padding.

Both datasets convert image patches into (coordinate, intensity) -> label pairs
suitable for training implicit neural representation models.
"""

import os

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class Patcher:
    """Utility for splitting images into non-overlapping patches and reassembling them.

    Handles images whose dimensions are not evenly divisible by the patch size
    by applying reflection padding before splitting. This ensures no information
    is lost at image boundaries.

    Args:
        patch_shape (tuple): Target patch size as (height, width).

    Example:
        >>> patcher = Patcher((256, 256))
        >>> patches, orig_shape = patcher.patch(image_tensor)  # Split
        >>> reconstructed = patcher.unpatch(patches, orig_shape)  # Reassemble
    """

    def __init__(self, patch_shape: tuple):
        assert len(patch_shape) == 2, "Only 2D patch shapes are supported."
        self.patch_shape = patch_shape

    def patch(self, data: torch.Tensor):
        """Split a single image into non-overlapping patches.

        Args:
            data: Image tensor of shape (C, H, W).

        Returns:
            patches: Tensor of shape (num_patches, C, patch_H, patch_W).
            original_shape: Tuple (H, W) before padding, needed for unpatch().
        """
        assert data.ndim == 3, "Input must be 3D: (channels, height, width)."
        channels, height, width = data.shape
        patch_h, patch_w = self.patch_shape

        # Compute padding to make dimensions divisible by patch size
        pad_h, pad_w = self._get_padding((height, width))
        padded = F.pad(data, (0, pad_w, 0, pad_h), mode="reflect")

        # Use unfold to extract non-overlapping patches
        patches = F.unfold(
            padded.unsqueeze(0), kernel_size=self.patch_shape, stride=self.patch_shape
        )
        patches = patches.reshape(channels, patch_h, patch_w, -1).permute(3, 0, 1, 2)

        return patches, (height, width)

    def unpatch(self, patches: torch.Tensor, original_shape: tuple) -> torch.Tensor:
        """Reconstruct the original image from patches.

        Args:
            patches: Tensor of shape (num_patches, C, patch_H, patch_W).
            original_shape: Original (H, W) before padding.

        Returns:
            Reconstructed image tensor of shape (C, H, W).
        """
        height, width = original_shape
        pad_h, pad_w = self._get_padding((height, width))
        padded_shape = (height + pad_h, width + pad_w)

        num_patches, channels, _, _ = patches.shape
        patches = patches.permute(1, 2, 3, 0).reshape(1, -1, num_patches)

        padded_image = F.fold(
            patches, output_size=padded_shape,
            kernel_size=self.patch_shape, stride=self.patch_shape,
        )

        # Remove padding to restore original dimensions
        return padded_image[0, :, :height, :width]

    def _get_padding(self, spatial_shape: tuple) -> tuple:
        """Calculate padding needed for even patch division."""
        h, w = spatial_shape
        ph, pw = self.patch_shape
        pad_h = (ph - h % ph) if h % ph != 0 else 0
        pad_w = (pw - w % pw) if w % pw != 0 else 0
        return pad_h, pad_w


class FIVESDataset(Dataset):
    """FIVES fundus image dataset for binary vessel segmentation.

    Splits high-resolution 2048x2048 fundus images into fixed-size patches
    and generates per-pixel (coordinate, intensity) pairs for INR training.

    Each sample returns:
        - coords_intensities: (patch_H * patch_W, 3) tensor of [x_norm, y_norm, intensity]
        - labels: (patch_H * patch_W,) tensor of binary vessel labels {0, 1}

    Args:
        images_path (str): Path to directory containing fundus images.
        masks_path (str): Path to directory containing binary vessel masks.
        patch_size (int): Size of square patches (default: 256).
        augment (bool): Whether to apply data augmentation (default: True).
        full_image_size (int): Full image dimension (default: 2048 for FIVES).
    """

    VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

    def __init__(
        self,
        images_path: str,
        masks_path: str,
        patch_size: int = 256,
        augment: bool = True,
        full_image_size: int = 2048,
    ):
        self.images_path = images_path
        self.masks_path = masks_path
        self.patch_size = patch_size
        self.full_image_size = full_image_size

        # List and sort valid image files, excluding hidden files (e.g., .DS_Store)
        self.image_files = sorted([
            f for f in os.listdir(images_path)
            if not f.startswith(".") and f.lower().endswith(self.VALID_EXTENSIONS)
        ])
        self.mask_files = sorted([
            f for f in os.listdir(masks_path)
            if not f.startswith(".") and f.lower().endswith(self.VALID_EXTENSIONS)
        ])

        self.augment = augment

        # Augmentation pipeline for medical image segmentation
        # Uses mild transformations to avoid distorting vessel structures
        self.augmentations = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussianBlur(blur_limit=3, p=0.2),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.CLAHE(clip_limit=2, p=0.3),  # Contrast-limited adaptive histogram eq.
            ],
            additional_targets={"mask": "image"},
        )

    def __len__(self):
        """Total number of patches across all images."""
        patches_per_image = (self.full_image_size // self.patch_size) ** 2
        return len(self.image_files) * patches_per_image

    def __getitem__(self, idx):
        """Load one patch and return (coords_intensities, labels).

        The idx encodes both which image and which patch within that image:
            image_idx = idx // patches_per_image
            patch_idx = idx % patches_per_image
        """
        patches_per_image = (self.full_image_size // self.patch_size) ** 2
        image_idx = idx // patches_per_image
        patch_idx = idx % patches_per_image

        # Load full grayscale image and mask
        image_path = os.path.join(self.images_path, self.image_files[image_idx])
        mask_path = os.path.join(self.masks_path, self.mask_files[image_idx])

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8) / 255

        # Compute patch position in the grid
        grid_cols = self.full_image_size // self.patch_size
        row = patch_idx // grid_cols
        col = patch_idx % grid_cols
        y_start, x_start = row * self.patch_size, col * self.patch_size
        y_end, x_end = y_start + self.patch_size, x_start + self.patch_size

        # Extract patch
        image_patch = image[y_start:y_end, x_start:x_end]
        mask_patch = mask[y_start:y_end, x_start:x_end]

        # Apply augmentation (jointly to image and mask)
        if self.augment:
            augmented = self.augmentations(image=image_patch, mask=mask_patch)
            image_patch, mask_patch = augmented["image"], augmented["mask"]

        # Convert to (coordinate, intensity) representation for INR
        coords_intensities = self._generate_coords_intensities(image_patch)
        labels = torch.tensor(mask_patch.flatten(), dtype=torch.long)

        return coords_intensities, labels

    def _generate_coords_intensities(self, image_patch: np.ndarray) -> torch.Tensor:
        """Convert an image patch to normalized (x, y, intensity) vectors.

        Creates a meshgrid of pixel coordinates normalized to [0, 1], then
        concatenates each pixel's intensity value as a third feature.

        Args:
            image_patch: 2D numpy array of shape (H, W) with values in [0, 1].

        Returns:
            Tensor of shape (H*W, 3) with columns [x_norm, y_norm, intensity].
        """
        if isinstance(image_patch, torch.Tensor):
            image_patch = image_patch.cpu().numpy()

        # Handle multi-channel images by converting to grayscale
        if image_patch.ndim == 3:
            if image_patch.shape[0] in (1, 3):
                image_patch = np.mean(image_patch, axis=0)

        H, W = image_patch.shape

        # Create normalized coordinate grid
        y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        coords = np.stack((x_coords, y_coords), axis=-1).reshape(-1, 2).astype(np.float32)
        coords /= np.array([max(W - 1, 1), max(H - 1, 1)], dtype=np.float32)

        # Concatenate coordinates with pixel intensities
        intensities = image_patch.flatten().reshape(-1, 1)
        coords_intensities = np.concatenate([coords, intensities], axis=-1)

        return torch.tensor(coords_intensities, dtype=torch.float32)


class RAVIRDataset(Dataset):
    """RAVIR infrared angiography dataset for multi-class vessel segmentation.

    Supports 3-class segmentation with dynamic patching via reflection padding.
    Mask values are mapped: 0 -> background, 128 -> vein (class 1), 255 -> artery (class 2).

    Each sample returns all patches from one image concatenated together:
        - coords_intensities: (total_pixels, 3) tensor
        - labels: (total_pixels,) tensor with values in {0, 1, 2}

    Args:
        images_path (str): Path to image directory.
        masks_path (str): Path to mask directory.
        target_size (tuple or None): Resize dimensions (H, W). None to keep original.
        patch_size (tuple): Patch dimensions for the Patcher (default: (258, 258)).
        augment (bool): Whether to apply augmentation.
    """

    # Mapping from raw mask pixel values to class indices
    MASK_VALUE_TO_CLASS = {0: 0, 128: 1, 255: 2}

    def __init__(
        self,
        images_path: str,
        masks_path: str,
        target_size: tuple = None,
        patch_size: tuple = (258, 258),
        augment: bool = True,
    ):
        self.images_path = images_path
        self.masks_path = masks_path
        self.image_files = sorted(os.listdir(images_path))
        self.mask_files = sorted(os.listdir(masks_path))
        self.target_size = target_size
        self.patch_size = patch_size
        self.augment = augment

        self.patcher = Patcher(self.patch_size)

        # Lighter augmentation for IR images (more sensitive to noise)
        self.augmentations = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=5, p=0.5),
                A.RandomBrightnessContrast(p=0.1),
                A.GaussianBlur(blur_limit=1, p=0.1),
                A.GaussNoise(var_limit=(5.0, 20.0), p=0.1),
                A.CLAHE(clip_limit=1, p=0.1),
            ],
            additional_targets={"mask": "image"},
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load grayscale image and mask
        image_path = os.path.join(self.images_path, self.image_files[idx])
        mask_path = os.path.join(self.masks_path, self.mask_files[idx])
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Optional resize
        if self.target_size is not None:
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        # Apply augmentation
        if self.augment:
            augmented = self.augmentations(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        # Normalize image and map mask values to class indices
        image = image.astype(np.float32) / 255.0
        mask = self._transform_mask(mask).astype(np.float32)

        # Convert to tensors with channel dimension
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        # Split into patches using reflection padding
        patches_image, _ = self.patcher.patch(image_tensor)
        patches_mask, _ = self.patcher.patch(mask_tensor)

        # Generate (coordinate, intensity) pairs for each patch
        coords_list, labels_list = [], []
        for img_patch, mask_patch in zip(patches_image, patches_mask):
            ci, lab = self._generate_coords_intensities_and_labels(img_patch, mask_patch)
            coords_list.append(ci)
            labels_list.append(lab)

        coords_intensities = torch.cat(coords_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        return coords_intensities, labels

    def _transform_mask(self, mask: np.ndarray) -> np.ndarray:
        """Map raw pixel values {0, 128, 255} to class indices {0, 1, 2}."""
        transformed = np.zeros_like(mask)
        for pixel_val, class_idx in self.MASK_VALUE_TO_CLASS.items():
            transformed[mask == pixel_val] = class_idx
        return transformed

    @staticmethod
    def _generate_coords_intensities_and_labels(
        image_patch: torch.Tensor, mask_patch: torch.Tensor
    ):
        """Convert a single patch pair to coordinate-intensity vectors and labels.

        Args:
            image_patch: Tensor of shape (1, H, W).
            mask_patch: Tensor of shape (1, H, W) with class indices.

        Returns:
            coords_intensities: (H*W, 3) tensor of [x_norm, y_norm, intensity].
            labels: (H*W,) tensor of class indices.
        """
        _, H, W = image_patch.shape

        # Create normalized coordinate grid
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H), torch.arange(W), indexing="ij"
        )
        coords = torch.stack((x_coords, y_coords), dim=-1).view(-1, 2).float()
        coords = coords / torch.tensor([max(W, 1), max(H, 1)], dtype=torch.float32)

        # Extract pixel intensities and labels
        intensities = image_patch.view(-1, 1)
        labels = mask_patch.view(-1)

        coords_intensities = torch.cat([coords, intensities], dim=-1)
        return coords_intensities, labels


def ravir_collate_fn(batch):
    """Custom collate function for RAVIRDataset.

    Since each image may produce a different number of total pixels
    (due to varying image sizes or patch counts), we concatenate rather
    than stack the batch elements.

    Args:
        batch: List of (coords_intensities, labels) tuples.

    Returns:
        coords_intensities: (total_pixels, 3) tensor.
        labels: (total_pixels,) tensor.
    """
    coords_list, labels_list = zip(*batch)
    return torch.cat(coords_list, dim=0), torch.cat(labels_list, dim=0)

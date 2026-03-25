"""
Evaluation and visualization utilities for INR-based retinal segmentation.

Provides functions for:
- Patch-based inference on full-resolution images
- Segmentation mask visualization with color-coded classes
- Metric computation (loss, qualitative comparison plots)
"""

import os

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from .datasets import Patcher


# ---------------------------------------------------------------------------
# Patch splitting and reconstruction (for binary / FIVES evaluation)
# ---------------------------------------------------------------------------

def split_into_patches(image: torch.Tensor, patch_size: int) -> list:
    """Split a single-channel image tensor into non-overlapping square patches.

    Args:
        image: Tensor of shape (1, H, W) — single channel image.
        patch_size: Side length of each square patch.

    Returns:
        List of tensors, each of shape (1, patch_size, patch_size).
    """
    _, H, W = image.shape
    patches = []
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            patch = image[:, i : i + patch_size, j : j + patch_size]
            patches.append(patch)
    return patches


def reconstruct_from_patches(
    patches: list, image_shape: tuple, patch_size: int
) -> torch.Tensor:
    """Reconstruct the full image from a list of predicted patches.

    Patches are placed back into their grid positions. Overlapping regions
    (if any) are averaged.

    Args:
        patches: List of tensors, each (1, pH, pW).
        image_shape: Target (H, W) of the reconstructed image.
        patch_size: Side length used during splitting.

    Returns:
        Reconstructed 2D tensor of shape (H, W).
    """
    H, W = image_shape
    device = patches[0].device
    reconstructed = torch.zeros((H, W), device=device)
    count_map = torch.zeros((H, W), device=device)

    idx = 0
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            ph, pw = patches[idx].shape[1:]
            reconstructed[i : i + ph, j : j + pw] += patches[idx][0]
            count_map[i : i + ph, j : j + pw] += 1
            idx += 1

    # Average overlapping regions
    count_map = torch.clamp(count_map, min=1)
    return reconstructed / count_map


# ---------------------------------------------------------------------------
# Binary evaluation (FIVES)
# ---------------------------------------------------------------------------

def evaluate_binary_model(
    model,
    dataset,
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int,
    criterion,
    device: str = "cpu",
):
    """Evaluate a binary segmentation model on a single test image.

    Splits the image into patches, runs inference per patch, reconstructs
    the full prediction, and displays a side-by-side comparison.

    Args:
        model: Trained INRSegmentationModel.
        dataset: Dataset instance (used for coordinate generation).
        image: Grayscale test image as numpy array (H, W), values in [0, 255].
        mask: Ground truth binary mask (H, W).
        patch_size: Patch size for inference.
        criterion: Loss function for evaluation.
        device: Compute device ('cpu', 'cuda', 'mps').

    Returns:
        predicted_mask: Binary numpy array (H, W) with values {0, 1}.
        total_loss: Sum of per-patch losses.
    """
    model.eval()

    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    image_tensor = image_tensor.squeeze(0)  # (1, H, W)

    patches = split_into_patches(image_tensor, patch_size)

    predicted_patches = []
    losses = []

    for img_patch in patches:
        H, W = img_patch.shape[1:]
        coords_intensities = dataset._generate_coords_intensities(
            img_patch.squeeze(0).cpu().numpy()
        ).to(device)

        with torch.no_grad():
            outputs = model(coords_intensities)
            outputs = outputs.squeeze(-1).view(H, W)
            predicted_labels = (outputs > 0.5).long()
            predicted_patches.append(predicted_labels.unsqueeze(0))

            targets = img_patch.reshape(-1)
            loss = criterion(outputs.view(-1), targets)
            losses.append(loss.item())

    predicted_mask = reconstruct_from_patches(
        predicted_patches, image_tensor.shape[1:], patch_size
    ).cpu().numpy()
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)
    total_loss = sum(losses)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(predicted_mask, cmap="gray")
    axes[2].set_title(f"Prediction (Loss: {total_loss:.4f})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    return predicted_mask, total_loss


# ---------------------------------------------------------------------------
# Multi-class evaluation (RAVIR)
# ---------------------------------------------------------------------------

# Color mapping for 3-class RAVIR segmentation
RAVIR_COLORS = {
    0: [255, 255, 255],  # Background: white
    1: [0, 0, 255],      # Vein: blue
    2: [255, 0, 0],      # Artery: red
}


def visualize_segmentation_mask(
    mask: np.ndarray, title: str = "Segmentation Mask", ax=None
):
    """Render a multi-class segmentation mask with color-coded classes.

    Args:
        mask: 2D numpy array (H, W) with integer class indices.
        title: Plot title.
        ax: Optional matplotlib axes. If None, creates a new figure.
    """
    H, W = mask.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for class_idx, color in RAVIR_COLORS.items():
        rgb[mask == class_idx] = color

    if ax is None:
        plt.figure(figsize=(6, 6))
        plt.imshow(rgb / 255.0)
        plt.title(title)
        plt.axis("off")
        plt.show()
    else:
        ax.imshow(rgb / 255.0)
        ax.set_title(title)
        ax.axis("off")


def generate_coords_intensities(image_patch: torch.Tensor, device: str = "cpu"):
    """Generate coordinate-intensity pairs for a single image patch.

    Standalone version for use during inference (not tied to a Dataset class).

    Args:
        image_patch: Tensor of shape (1, H, W) or (H, W).
        device: Target device for the output tensor.

    Returns:
        Tensor of shape (H*W, 3) with [x_norm, y_norm, intensity].
    """
    if image_patch.dim() == 2:
        image_patch = image_patch.unsqueeze(0)

    C, H, W = image_patch.shape
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
    )
    coords = torch.stack((x_coords, y_coords), dim=-1).view(-1, 2).float()
    coords = coords / torch.tensor([max(W - 1, 1), max(H - 1, 1)], dtype=torch.float32, device=device)

    intensities = image_patch.view(-1).to(device).unsqueeze(-1)
    return torch.cat([coords, intensities], dim=-1)


def evaluate_multiclass_model(
    model,
    image: np.ndarray,
    img_size: tuple,
    patch_size: tuple = (258, 258),
    device: str = "cpu",
    ground_truth_mask: np.ndarray = None,
    criterion=None,
):
    """Evaluate a multi-class segmentation model on a single RAVIR test image.

    Uses the Patcher to split the image, runs inference, reconstructs the
    prediction, and optionally computes loss against ground truth.

    Args:
        model: Trained INRSegmentationModel (num_classes=3).
        image: Grayscale image as numpy array (H, W), values in [0, 255].
        img_size: Tuple (H, W) for reconstruction.
        patch_size: Patch dimensions for splitting.
        device: Compute device.
        ground_truth_mask: Optional GT mask (H, W) with class indices {0,1,2}.
        criterion: Optional loss function for quantitative evaluation.

    Returns:
        predicted_mask: Numpy array (H, W) with class indices.
        loss_value: Loss against ground truth (or None if GT not provided).
    """
    model.eval()
    patcher = Patcher(patch_size)

    # Normalize and convert to tensor
    image_normalized = image.astype(np.float32) / 255.0
    image_tensor = torch.tensor(image_normalized, dtype=torch.float32).unsqueeze(0).to(device)

    # Split into patches
    patches_image, _ = patcher.patch(image_tensor)

    # Generate coordinate-intensity pairs for each patch
    coords_list, patch_sizes = [], []
    for img_patch in patches_image:
        if img_patch.dim() == 2:
            img_patch = img_patch.unsqueeze(0)
        ci = generate_coords_intensities(img_patch, device=device)
        coords_list.append(ci)
        patch_sizes.append((img_patch.shape[1], img_patch.shape[2]))

    all_coords = torch.cat(coords_list, dim=0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(all_coords)
        predicted_labels = outputs.argmax(dim=-1)

    # Reconstruct from per-pixel predictions
    start = 0
    predicted_patches = []
    for (H, W) in patch_sizes:
        n_pixels = H * W
        patch_labels = predicted_labels[start : start + n_pixels].view(H, W)
        predicted_patches.append(patch_labels)
        start += n_pixels

    predicted_tensor = torch.stack(predicted_patches).unsqueeze(1).float()
    predicted_mask = patcher.unpatch(predicted_tensor, img_size)
    predicted_mask = predicted_mask.long().cpu().numpy().astype(np.uint8).squeeze()

    # Compute loss if ground truth is provided
    loss_value = None
    if ground_truth_mask is not None and criterion is not None:
        gt_tensor = torch.tensor(ground_truth_mask, dtype=torch.long).view(-1).to(device)
        loss_value = criterion(outputs, gt_tensor).item()

    # Visualization
    fig, axes = plt.subplots(1, 2 if ground_truth_mask is None else 3, figsize=(18, 6))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    title = "Predicted Mask" if loss_value is None else f"Predicted Mask (Loss: {loss_value:.4f})"
    visualize_segmentation_mask(predicted_mask, title=title, ax=axes[1])

    if ground_truth_mask is not None:
        visualize_segmentation_mask(ground_truth_mask.astype(np.uint8), title="Ground Truth", ax=axes[2])

    plt.tight_layout()
    plt.show()

    return predicted_mask, loss_value

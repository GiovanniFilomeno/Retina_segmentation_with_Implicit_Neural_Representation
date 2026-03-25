"""
Retinal Vessel Segmentation with Implicit Neural Representations (INR).

This package provides modular components for training and evaluating
INR-based segmentation models on retinal fundus and IR angiography images.

Modules:
    models   - Neural network architectures (SIREN, Positional Encoding, INR)
    datasets - Dataset classes for FIVES and RAVIR benchmarks
    losses   - Focal-Dice and multi-class segmentation losses
    utils    - Evaluation, visualization, and patch reconstruction utilities
"""

from .models import INRSegmentationModel, SineLayer, PositionalEncoding, AdaptiveDropout
from .datasets import FIVESDataset, RAVIRDataset, Patcher
from .losses import FocalDiceLoss, BinaryFocalDiceLoss
from .utils import (
    evaluate_binary_model,
    evaluate_multiclass_model,
    visualize_segmentation_mask,
    split_into_patches,
    reconstruct_from_patches,
)

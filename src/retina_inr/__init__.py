"""Reusable components for the retinal pointwise-INR research prototype.

This package exposes a coordinate-conditioned MLP, strict preprocessing,
logit-based losses, reconstruction-safe patch inference, and per-image metrics.
It makes no claim that the pointwise architecture observes spatial context:
coordinates are patch-local and each pixel is classified independently.
"""

from .datasets import (
    DEFAULT_MASK_SUFFIXES,
    RAVIR_VALUE_TO_CLASS,
    FIVESDataset,
    ImageMaskPair,
    Patcher,
    PatchMetadata,
    RAVIRDataset,
    RetinaPatchDataset,
    build_training_augmentation,
    concatenate_variable_length_batch,
    encode_binary_mask,
    encode_multiclass_mask,
    image_to_features,
    make_coordinate_grid,
    normalize_grayscale_image,
    pair_image_mask_files,
    ravir_collate_fn,
    validate_class_labels,
)
from .losses import (
    BinaryFocalDiceLoss,
    FocalDiceLoss,
    MulticlassFocalDiceLoss,
)
from .metrics import binary_segmentation_metrics, multiclass_segmentation_metrics
from .models import INRSegmentationModel, PositionalEncoding, SineLayer
from .utils import (
    EvaluationResult,
    chunked_forward,
    colorize_segmentation_mask,
    evaluate_binary_model,
    evaluate_multiclass_model,
    predict_image_logits,
)

__all__ = [
    "DEFAULT_MASK_SUFFIXES",
    "RAVIR_VALUE_TO_CLASS",
    "BinaryFocalDiceLoss",
    "EvaluationResult",
    "FIVESDataset",
    "FocalDiceLoss",
    "INRSegmentationModel",
    "ImageMaskPair",
    "MulticlassFocalDiceLoss",
    "PatchMetadata",
    "Patcher",
    "PositionalEncoding",
    "RAVIRDataset",
    "RetinaPatchDataset",
    "SineLayer",
    "binary_segmentation_metrics",
    "build_training_augmentation",
    "chunked_forward",
    "colorize_segmentation_mask",
    "concatenate_variable_length_batch",
    "encode_binary_mask",
    "encode_multiclass_mask",
    "evaluate_binary_model",
    "evaluate_multiclass_model",
    "image_to_features",
    "make_coordinate_grid",
    "multiclass_segmentation_metrics",
    "normalize_grayscale_image",
    "pair_image_mask_files",
    "predict_image_logits",
    "ravir_collate_fn",
    "validate_class_labels",
]

__version__ = "0.1.0"

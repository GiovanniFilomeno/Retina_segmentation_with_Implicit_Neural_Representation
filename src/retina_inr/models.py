"""Pointwise implicit models for retinal segmentation.

The model in this module is deliberately described narrowly: it is a
coordinate-conditioned, pointwise MLP.  Each prediction sees only the pixel's
patch-local ``(x, y)`` coordinates and grayscale intensity.  It does *not* see
neighbouring pixels and is therefore not a replacement for a convolutional or
attention-based image encoder.

All model outputs are logits.  Binary models emit exactly one logit per pixel;
multiclass models emit one logit per class.  Probability transforms belong in
the loss, metric, or presentation layer.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn

SegmentationTask = Literal["binary", "multiclass"]
ActivationName = Literal["sine", "relu"]


class PositionalEncoding(nn.Module):
    """Encode 2-D coordinates with dyadic Fourier features.

    Coordinates are expected in ``[-1, 1]``.  The original coordinates are
    retained alongside sine and cosine features, which makes low-frequency
    structure directly available to the MLP.
    """

    def __init__(self, num_freqs: int, *, input_dims: int = 2) -> None:
        super().__init__()
        if num_freqs < 0:
            raise ValueError("num_freqs must be non-negative")
        if input_dims < 1:
            raise ValueError("input_dims must be positive")

        self.num_freqs = int(num_freqs)
        self.input_dims = int(input_dims)
        frequencies = 2.0 ** torch.arange(self.num_freqs, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)

    @property
    def output_dim(self) -> int:
        """Number of features produced for one coordinate vector."""

        return self.input_dims * (1 + 2 * self.num_freqs)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.shape[-1] != self.input_dims:
            raise ValueError(
                f"expected coordinates with last dimension {self.input_dims}, "
                f"got shape {tuple(coordinates.shape)}"
            )
        if not coordinates.is_floating_point():
            raise TypeError("coordinates must be a floating-point tensor")
        if self.num_freqs == 0:
            return coordinates

        frequencies = self.frequencies.to(device=coordinates.device, dtype=coordinates.dtype)
        phases = math.pi * coordinates.unsqueeze(-1) * frequencies
        fourier = torch.cat((torch.sin(phases), torch.cos(phases)), dim=-1)
        fourier = fourier.flatten(start_dim=-2)
        return torch.cat((coordinates, fourier), dim=-1)


class SineLayer(nn.Module):
    """A SIREN linear layer followed by a sine activation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError("in_features and out_features must be positive")
        if omega_0 <= 0:
            raise ValueError("omega_0 must be positive")

        self.in_features = int(in_features)
        self.is_first = bool(is_first)
        self.omega_0 = float(omega_0)
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Apply the initialization proposed for sinusoidal networks."""

        if not hasattr(self, "linear"):
            return
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(inputs))


class INRSegmentationModel(nn.Module):
    """Coordinate-conditioned pointwise MLP that returns segmentation logits.

    Parameters
    ----------
    task:
        ``"binary"`` for a one-logit vessel/background model, or
        ``"multiclass"`` for a K-logit model.
    num_classes:
        Required and at least two for multiclass models.  For binary models it
        must be omitted (or set to one), because binary output is one logit,
        not two softmax scores.
    hidden_dim, num_layers:
        Width and number of hidden pointwise layers.
    num_freqs:
        Number of dyadic Fourier bands applied to patch-local coordinates.
    omega_0:
        Frequency scale used by sine hidden layers.
    activation:
        ``"sine"`` for the INR model or ``"relu"`` for a plain-MLP ablation.

    Notes
    -----
    Inputs have shape ``(..., 3)`` and columns ``[x, y, intensity]``.  The
    coordinate helpers in :mod:`retina_inr.datasets` generate ``x`` and ``y``
    in ``[-1, 1]`` independently within every patch and intensity in ``[0, 1]``.
    The output has shape ``(..., 1)`` for binary models and ``(..., K)`` for
    multiclass models.  No sigmoid or softmax is applied.
    """

    def __init__(
        self,
        *,
        task: SegmentationTask,
        num_classes: int | None = None,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_freqs: int = 6,
        omega_0: float = 30.0,
        dropout: float = 0.0,
        activation: ActivationName = "sine",
    ) -> None:
        super().__init__()
        if task not in ("binary", "multiclass"):
            raise ValueError("task must be 'binary' or 'multiclass'")
        if task == "binary":
            if num_classes not in (None, 1):
                raise ValueError("binary models emit one logit; omit num_classes or set it to 1")
            output_channels = 1
        else:
            if num_classes is None or num_classes < 2:
                raise ValueError("multiclass models require num_classes to be at least 2")
            output_channels = int(num_classes)

        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if activation not in ("sine", "relu"):
            raise ValueError("activation must be 'sine' or 'relu'")

        self.task = task
        self.num_classes = output_channels
        self.output_channels = output_channels
        self.activation_name = activation
        self.positional_encoding = PositionalEncoding(num_freqs, input_dims=2)

        input_dim = self.positional_encoding.output_dim + 1
        hidden_layers: list[nn.Module] = []
        for layer_index in range(num_layers):
            layer_input_dim = input_dim if layer_index == 0 else hidden_dim
            if activation == "sine":
                hidden_layers.append(
                    SineLayer(
                        layer_input_dim,
                        hidden_dim,
                        is_first=layer_index == 0,
                        omega_0=omega_0,
                    )
                )
            else:
                hidden_layers.extend(
                    (nn.Linear(layer_input_dim, hidden_dim), nn.ReLU(inplace=False))
                )
            if dropout > 0:
                hidden_layers.append(nn.Dropout(dropout))

        self.backbone = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(hidden_dim, output_channels)

    @classmethod
    def binary(cls, **kwargs: object) -> INRSegmentationModel:
        """Construct a binary one-logit model."""

        return cls(task="binary", **kwargs)

    @classmethod
    def multiclass(cls, num_classes: int, **kwargs: object) -> INRSegmentationModel:
        """Construct a multiclass K-logit model."""

        return cls(task="multiclass", num_classes=num_classes, **kwargs)

    def forward(self, pixel_features: torch.Tensor) -> torch.Tensor:
        if pixel_features.ndim < 2 or pixel_features.shape[-1] != 3:
            raise ValueError("pixel_features must have shape (..., 3) with [x, y, intensity]")
        if not pixel_features.is_floating_point():
            raise TypeError("pixel_features must be a floating-point tensor")

        coordinates = pixel_features[..., :2]
        intensity = pixel_features[..., 2:3]
        encoded = self.positional_encoding(coordinates)
        hidden = self.backbone(torch.cat((encoded, intensity), dim=-1))
        return self.output_layer(hidden)

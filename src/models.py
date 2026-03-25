"""
Neural network architectures for Implicit Neural Representation (INR) segmentation.

This module implements SIREN-based (Sinusoidal Representation Networks) architectures
for pixel-wise retinal vessel segmentation. Instead of operating on image patches with
convolutions, INR models learn a continuous function mapping (x, y, intensity) -> class,
enabling resolution-independent inference.

Key references:
    - Sitzmann et al., "Implicit Neural Representations with Periodic Activation
      Functions" (NeurIPS 2020)
    - Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields
      for View Synthesis" (ECCV 2020) — for positional encoding

Architecture overview:
    Input: (x, y, intensity) per pixel
        |
    [Positional Encoding] — maps 2D coordinates to high-frequency features
        |
    [Reduction Layer] — projects encoded features to hidden dimension
        |
    [SIREN MLP] — multiple sine-activated layers with batch normalization
        |
    Output: class probabilities per pixel
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Fourier feature positional encoding for 2D spatial coordinates.

    Maps low-dimensional coordinates (x, y) into a higher-dimensional space
    using sinusoidal functions at exponentially increasing frequencies. This
    allows the MLP to learn high-frequency spatial patterns that would
    otherwise be difficult to capture (the "spectral bias" problem).

    For each coordinate dimension and each frequency level k:
        output = [sin(2^0 * x), cos(2^0 * x), sin(2^1 * x), cos(2^1 * x), ...,
                  sin(2^(L-1) * x), cos(2^(L-1) * x)]

    Args:
        num_freqs (int): Number of frequency bands (L). Output dimension per
            coordinate = 2 * num_freqs. Total output dim = num_coords * 2 * num_freqs.
    """

    def __init__(self, num_freqs: int):
        super().__init__()
        self.num_freqs = num_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode coordinates with sinusoidal positional features.

        Args:
            x: Coordinate tensor of shape (N, num_coords), where N is the
               number of pixels and num_coords is typically 2 for (x, y).

        Returns:
            Encoded tensor of shape (N, num_coords * num_freqs * 2).
        """
        # Generate exponentially spaced frequencies: [2^0, 2^1, ..., 2^(L-1)]
        frequencies = torch.linspace(0, self.num_freqs - 1, self.num_freqs, device=x.device)
        frequencies = 2.0 ** frequencies  # Shape: (num_freqs,)
        frequencies = frequencies.view(1, 1, -1)  # Shape: (1, 1, num_freqs)

        # Broadcast multiply: each coordinate with each frequency
        x = x.unsqueeze(-1)  # Shape: (N, num_coords, 1)
        x = x * frequencies  # Shape: (N, num_coords, num_freqs)

        # Apply sin and cos, then concatenate along frequency dimension
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # (N, num_coords, 2*num_freqs)

        # Flatten to (N, num_coords * 2 * num_freqs)
        x = x.view(x.shape[0], -1)
        return x


class AdaptiveDropout(nn.Module):
    """Dropout with exponentially decaying probability during training.

    Starts with a higher dropout rate for strong regularization in early
    epochs, then gradually reduces it as the model converges. Call `step()`
    at the end of each epoch to decay the probability.

    Args:
        initial_p (float): Initial dropout probability (default: 0.5).
        decay_factor (float): Multiplicative decay per step (default: 0.95).
            After k steps, p = initial_p * decay_factor^k.
    """

    def __init__(self, initial_p: float = 0.5, decay_factor: float = 0.95):
        super().__init__()
        self.p = initial_p
        self.decay_factor = decay_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return F.dropout(x, p=self.p, training=True)
        return x

    def step(self):
        """Decay the dropout probability by one step."""
        self.p *= self.decay_factor


class SineLayer(nn.Module):
    """A single SIREN layer: linear transformation followed by sine activation.

    SIREN layers use sin(omega_0 * Wx + b) as activation, which is better suited
    for representing signals with fine spatial detail compared to ReLU. The weight
    initialization scheme is critical — it ensures that the distribution of
    activations is preserved through the network depth.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include bias in the linear layer.
        is_first (bool): If True, uses uniform initialization in [-1/in, 1/in].
            If False, uses the SIREN-specific initialization scaled by omega_0.
        omega_0 (float): Frequency scaling factor for the sine activation.
            Controls the bandwidth of the represented signal.
        is_last (bool): If True, applies softmax after sine (for multi-class output).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 1.0,
        is_last: bool = False,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.is_last = is_last
        self.in_features = in_features

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following the SIREN paper (Sitzmann et al., 2020).

        First layer: W ~ U(-1/n, 1/n) where n = in_features
        Other layers: W ~ U(-sqrt(6/n)/omega_0, sqrt(6/n)/omega_0)

        This ensures that the input to the sine activation has unit variance,
        preventing vanishing or exploding activations through deep networks.
        """
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = np.sqrt(6.0 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = torch.sin(self.omega_0 * x)
        if self.is_last:
            x = F.softmax(x, dim=-1)
        return x


class INRSegmentationModel(nn.Module):
    """Implicit Neural Representation model for pixel-wise vessel segmentation.

    This model takes per-pixel features (normalized coordinates + intensity) and
    predicts class probabilities. The architecture consists of:

    1. Positional Encoding: Maps (x, y) coordinates to high-dimensional Fourier features
    2. Reduction Layer: Linear projection to hidden dimension
    3. SIREN MLP: Multiple sine-activated hidden layers with batch normalization
    4. Output Layer: Projects to class probabilities

    The model is resolution-independent: trained on small patches (e.g., 256x256),
    it can perform inference at any resolution by simply querying with a denser
    coordinate grid.

    Args:
        num_classes (int): Number of segmentation classes.
            - 2 for binary (vessel vs. background) on FIVES
            - 3 for multi-class (background, vein, artery) on RAVIR
        hidden_dim (int): Width of hidden layers (default: 256).
        num_layers (int): Total number of MLP layers including first and last (default: 5).
        num_freqs (int): Number of positional encoding frequency bands (default: 10).
        initial_dropout_p (float): Starting probability for adaptive dropout.
        outermost_linear (bool): If True, the last layer is linear + softmax
            instead of a SineLayer. Can improve training stability.
        linear_network (bool): If True, uses ReLU-activated linear layers instead
            of sine layers for the intermediate blocks (ablation study).

    Input shape:  (N, 3) where columns are [x_norm, y_norm, intensity]
    Output shape: (N, num_classes) class probabilities per pixel
    """

    def __init__(
        self,
        num_classes: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 5,
        num_freqs: int = 10,
        initial_dropout_p: float = 0.5,
        outermost_linear: bool = False,
        linear_network: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.outermost_linear = outermost_linear

        # --- Positional encoding ---
        self.pos_enc = PositionalEncoding(num_freqs)
        num_coords = 2  # (x, y)
        input_dim = num_coords * num_freqs * 2 + 1  # +1 for pixel intensity

        # --- Dimensionality reduction ---
        # Project high-dimensional positional features to a fixed hidden size
        self.reduction_layer = nn.Linear(input_dim, hidden_dim)

        # --- Adaptive dropout for regularization ---
        self.dropouts = nn.ModuleList(
            [AdaptiveDropout(initial_dropout_p) for _ in range(num_layers - 1)]
        )

        # --- Build MLP backbone ---
        self.mlp = nn.ModuleList()

        # First hidden layer (uses is_first=True initialization)
        self.mlp.append(
            nn.Sequential(
                SineLayer(hidden_dim, hidden_dim, is_first=True),
                nn.BatchNorm1d(hidden_dim),
            )
        )

        # Intermediate hidden layers
        for _ in range(1, num_layers - 2):
            if linear_network:
                # Ablation: standard ReLU layers instead of sine
                self.mlp.append(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden_dim),
                    )
                )
            else:
                self.mlp.append(
                    nn.Sequential(
                        SineLayer(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                    )
                )

        # Output layer
        if outermost_linear:
            self.mlp.append(nn.Linear(hidden_dim, num_classes))
            self.softmax = nn.Softmax(dim=-1)
        else:
            self.mlp.append(
                SineLayer(hidden_dim, num_classes, is_last=(num_classes > 2))
            )

    def forward(self, coords_intensities: torch.Tensor) -> torch.Tensor:
        """Forward pass: map per-pixel features to class probabilities.

        Args:
            coords_intensities: Tensor of shape (N, 3) where each row is
                [x_normalized, y_normalized, pixel_intensity].

        Returns:
            Tensor of shape (N, num_classes) with class probabilities,
            or (N, 1) with sigmoid probability for binary segmentation.
        """
        # Split input into spatial coordinates and intensity
        coords = coords_intensities[:, :-1]  # (N, 2)
        intensities = coords_intensities[:, -1].unsqueeze(-1)  # (N, 1)

        # Apply positional encoding to spatial coordinates
        x = self.pos_enc(coords)  # (N, num_coords * num_freqs * 2)

        # Concatenate encoded coordinates with raw intensity
        x = torch.cat([x, intensities], dim=-1)  # (N, input_dim)

        # Project to hidden dimension
        x = self.reduction_layer(x)  # (N, hidden_dim)

        # Pass through SIREN MLP with dropout
        for i, layer in enumerate(self.mlp[:-1]):
            x = layer(x)
            x = self.dropouts[i](x)

        # Final output layer
        x = self.mlp[-1](x)

        # Apply appropriate output activation
        if self.outermost_linear:
            x = self.softmax(x)
        elif self.num_classes <= 2 and not isinstance(self.mlp[-1], SineLayer):
            x = torch.sigmoid(x)

        return x

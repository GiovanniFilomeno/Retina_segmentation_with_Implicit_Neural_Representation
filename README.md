# Retinal Vessel Segmentation with Implicit Neural Representations

<p align="center">
  <b>Resolution-independent retinal vessel segmentation using SIREN-based Implicit Neural Representations</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/framework-PyTorch-red" alt="PyTorch">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/datasets-FIVES%20%7C%20RAVIR-orange" alt="Datasets">
</p>

---

## Overview

This project explores **Implicit Neural Representations (INR)** as an alternative to convolutional neural networks for retinal vessel segmentation. Instead of learning convolution filters over image patches, the model learns a **continuous function** mapping pixel coordinates and intensity to segmentation labels:

$$f_\theta: (x, y, I) \longrightarrow \text{class probability}$$

This formulation is inherently **resolution-independent** — a model trained on 256×256 patches can perform inference at any resolution by simply varying the density of the query coordinate grid.

### Key Contributions

- **SIREN-based architecture** with Fourier positional encoding for medical image segmentation
- **Focal-Dice combined loss** to handle severe class imbalance (vessels = 5–15% of pixels)
- **Patch-based training** with high-variance patch selection for efficient learning
- Evaluation on two complementary benchmarks: **FIVES** (binary, fundus) and **RAVIR** (3-class, IR angiography)
- **Ablation studies** on network depth, activation functions, loss functions, and learning rate schedules

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INR Segmentation Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input (per pixel)     Positional        SIREN MLP              │
│  ┌───────────────┐     Encoding          ┌──────────────┐       │
│  │ x_norm ∈ [0,1]│─┐   ┌──────────┐  ┌─▶│ SineLayer    │       │
│  │ y_norm ∈ [0,1]│─┼──▶│ sin/cos  │──┤  │ + BatchNorm  │×(L-2) │
│  │ intensity     │─┘   │ 2^k freq │  │  └──────┬───────┘       │
│  └───────────────┘     └──────────┘  │         │               │
│                              │       │         ▼               │
│                        [Concatenate]  │  ┌──────────────┐       │
│                              │       │  │ Output Layer  │       │
│                        [Reduction]───┘  │ → σ / softmax │       │
│                        Linear(→H)       └──────┬───────┘       │
│                                                │               │
│                                                ▼               │
│                                         class probabilities     │
└─────────────────────────────────────────────────────────────────┘
```

**Positional Encoding** maps 2D coordinates to high-frequency Fourier features, overcoming the spectral bias of MLPs:

$$\gamma(x) = \bigl[\sin(2^0 x),\; \cos(2^0 x),\; \sin(2^1 x),\; \cos(2^1 x),\; \dots,\; \sin(2^{L-1} x),\; \cos(2^{L-1} x)\bigr]$$

**SIREN Layers** use sinusoidal activation $h_{i+1} = \sin(\omega_0 \cdot W_i h_i + b_i)$ with a specialized weight initialization (Sitzmann et al., NeurIPS 2020) that preserves activation variance through depth.

---

## Datasets

| Dataset | Modality | Resolution | Classes | Task |
|---------|----------|------------|---------|------|
| [**FIVES**](https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169) | Color Fundus | 2048 × 2048 | 2 | Binary vessel segmentation |
| [**RAVIR**](https://ravir.grand-challenge.org/) | IR Angiography | 768 × 768 | 3 | Vein / Artery / Background |

---

## Results

### FIVES — Binary Vessel Segmentation

| Model | Layers | α | γ | Focal-Dice Loss |
|-------|--------|---|---|-----------------|
| SIREN + FocalDice | **4** | 0.75 | 2 | **0.8846** |
| SIREN + FocalDice | 6 | 0.75 | 2 | 0.9217 |
| SIREN + FocalDice | 2 | 0.75 | 2 | 0.8762 |

### RAVIR — Multi-Class Segmentation (Ablation Study)

| Layers | Output | α | γ | Freq | Batch | Loss | Notes |
|--------|--------|---|---|------|-------|------|-------|
| 6 | Linear+Softmax | 0.8 | 2 | 10 | 16 | 1.249 | LR=1e-3 best |
| 9 | Sine+Softmax | 0.8 | 2 | 10 | 16 | **0.987** | Best convergence |
| 9 | Sine+Softmax | 0.8 | 2 | 10 | 16 | 0.910 | + Dropout → degenerates |
| 3 | Linear+Softmax | 0.5 | 0.5 | 30 | 16 | 1.187 | Higher freq helps |

**Key Findings:**
- 4-layer SIREN achieves optimal depth–performance tradeoff on FIVES
- Sine activation consistently outperforms ReLU for fine vessel structures
- Focal-Dice loss is critical for handling the ~85/15% class imbalance
- Dropout with sine activations can cause degenerate predictions — use with caution
- Learning rate 1e-3 with ReduceLROnPlateau provides stable convergence

---

## Project Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/                          # Reusable Python package
│   ├── __init__.py
│   ├── models.py                 # INR model, SIREN layers, positional encoding
│   ├── datasets.py               # FIVES & RAVIR dataset classes, Patcher
│   ├── losses.py                 # Focal-Dice loss (binary & multi-class)
│   └── utils.py                  # Evaluation, visualization, patch utilities
│
├── notebooks/                    # Main experiment notebooks (start here)
│   ├── 01_fives_binary_segmentation.ipynb
│   ├── 02_ravir_multiclass_segmentation.ipynb
│   └── 03_ravir_patch_generation.ipynb
│
├── checkpoints/                  # Trained model weights (.pth)
│
├── data/                         # Datasets (not tracked — see data/README.md)
│   ├── README.md                 # Download instructions
│   ├── FIVES/
│   └── RAVIR/
│
├── experiments/                  # Archived exploratory notebooks
│   ├── fives_original.ipynb
│   ├── ravir_original.ipynb
│   └── old/                      # Early prototypes and ablations
│
└── references/                   # Third-party code and papers
    ├── liif/                     # LIIF reference implementation
    └── literature/
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/retinal-vessel-segmentation-inr.git
cd retinal-vessel-segmentation-inr

# Option A: conda
conda env create -f environment.yml
conda activate retina-inr

# Option B: pip
pip install -r requirements.txt
```

### 2. Download Datasets

Follow the instructions in [`data/README.md`](data/README.md) to download and place the
FIVES and RAVIR datasets.

### 3. Run Notebooks

Open the notebooks in order:

| Notebook | Description |
|----------|-------------|
| [`01_fives_binary_segmentation.ipynb`](notebooks/01_fives_binary_segmentation.ipynb) | Binary vessel segmentation on FIVES (2048×2048 fundus images) |
| [`02_ravir_multiclass_segmentation.ipynb`](notebooks/02_ravir_multiclass_segmentation.ipynb) | 3-class segmentation on RAVIR (IR angiography) |
| [`03_ravir_patch_generation.ipynb`](notebooks/03_ravir_patch_generation.ipynb) | Utility: extract high-variance training patches |

### 4. Use as a Python Package

```python
from src import INRSegmentationModel, FIVESDataset, BinaryFocalDiceLoss

# Initialize model
model = INRSegmentationModel(
    num_classes=2,
    hidden_dim=256,
    num_layers=4,
    num_freqs=5,
)

# Load trained weights
model.load_state_dict(torch.load("checkpoints/fives_binary_4layers.pth"))
```

---

## Technical Deep Dive

### Why INR Instead of CNNs?

| Property | CNN (U-Net, etc.) | INR (This Work) |
|----------|------------------|-----------------|
| Input | Fixed-size image patches | Per-pixel (x, y, intensity) |
| Resolution | Tied to training resolution | **Resolution-independent** |
| Parameters | ~31M (U-Net) | **~0.5M** |
| Inductive bias | Translation equivariance | Continuous signal representation |
| Inference | Requires interpolation for upscaling | Natively arbitrary resolution |

### Loss Function Design

The **Focal-Dice Loss** combines two complementary objectives:

**Focal Loss** — Addresses class imbalance by down-weighting well-classified pixels:
$$\mathcal{L}_{\text{focal}} = -\alpha (1-p)^\gamma \log(p)$$

**Dice Loss** — Directly optimizes spatial overlap (F1 score):
$$\mathcal{L}_{\text{dice}} = 1 - \frac{2|A \cap B| + \epsilon}{|A| + |B| + \epsilon}$$

### SIREN Weight Initialization

Critical for stable training of deep sine networks:
- **First layer**: $W \sim \mathcal{U}\left(-\frac{1}{n}, \frac{1}{n}\right)$
- **Other layers**: $W \sim \mathcal{U}\left(-\frac{\sqrt{6/n}}{\omega_0}, \frac{\sqrt{6/n}}{\omega_0}\right)$

This ensures unit-variance activations, preventing vanishing/exploding gradients.

---

## References

1. **Sitzmann, V.** et al. *Implicit Neural Representations with Periodic Activation Functions.* NeurIPS 2020. [[paper](https://arxiv.org/abs/2006.09661)]
2. **Mildenhall, B.** et al. *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis.* ECCV 2020. [[paper](https://arxiv.org/abs/2003.08934)]
3. **Chen, Y.** et al. *Learning Continuous Image Representation with Local Implicit Image Function.* CVPR 2021. [[paper](https://arxiv.org/abs/2012.09161)]
4. **Lin, T.-Y.** et al. *Focal Loss for Dense Object Detection.* ICCV 2017. [[paper](https://arxiv.org/abs/1708.02002)]
5. **Jin, K.** et al. *FIVES: A Fundus Image Dataset for AI-based Vessel Segmentation.* Scientific Data, 2022. [[paper](https://doi.org/10.1038/s41597-022-01564-3)]

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
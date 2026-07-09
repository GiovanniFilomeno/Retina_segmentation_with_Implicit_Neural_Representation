# Reproducibility guide

This guide separates two different goals:

1. **software reproducibility** — rebuilding the maintained package and exercising
   its contracts on synthetic inputs; and
2. **scientific reproducibility** — recreating an original training run and its
   reported held-out metrics.

The first is supported. The second is not claimed because the required original run
artifacts were not retained.

## Software verification

### Supported environment

- Python 3.10 or newer
- CPU is sufficient for all synthetic checks
- PyTorch and package dependencies declared in `pyproject.toml`

From a clean clone:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

python -m retina_inr.smoke
pytest
```

The smoke command uses generated tensors only. It checks the binary and multiclass
model interfaces and fails fast when a core numerical or shape invariant is broken.
It does not download data or evaluate segmentation quality.

The unexecuted notebook
[`notebooks/01_synthetic_smoke_test.ipynb`](../notebooks/01_synthetic_smoke_test.ipynb)
provides the same CPU-oriented entry point for readers who prefer Jupyter. Its empty
outputs are intentional: the repository does not present locally generated console
text as research evidence.

## Data acquisition

Data is deliberately excluded from version control. Follow
[`data/README.md`](../data/README.md), retain the original archive names, and record:

- source URL and retrieval date;
- dataset release/version, if supplied;
- archive and extracted-file checksums;
- applicable license or data-use agreement; and
- any local renaming, filtering, or conversion.

Do not commit medical images, derived patches, annotations, or identifiers unless the
dataset terms explicitly permit redistribution and the repository owner has approved
the change.

## Configuration and random state

Configuration examples describe explicit model and preprocessing parameters, but
they are not records of certified historical runs. For new experiments:

1. copy a config rather than editing it in place;
2. store it with the run artifacts;
3. set and record Python, NumPy, and PyTorch random seeds;
4. record deterministic-backend settings and any nondeterministic operations;
5. capture the exact package version or Git commit; and
6. log the operating system, Python/PyTorch/CUDA versions, GPU model, and driver.

Deterministic flags can reduce performance and do not guarantee bitwise equivalence
across PyTorch releases or hardware. Report that boundary rather than promising it.

## Split and patch provenance

Patch-based medical-imaging experiments are vulnerable to leakage. Split complete
source images—and patients, where identifiers permit—before creating patches. A
reproducible run should retain a machine-readable manifest containing at least:

```text
dataset_release, source_image_id, source_mask_id, split,
patch_origin_y, patch_origin_x, patch_height, patch_width,
preprocessing_config, source_checksum
```

Image/mask pairs should be joined by validated identifiers, not by independently
sorted directory listings. Geometric transforms must be shared between image and
mask; photometric transforms must not modify categorical masks. Masks must use
nearest-neighbour interpolation.

## Evaluation protocol

A benchmark report should be generated from held-out predictions, not transcribed
from a training notebook. At minimum, preserve:

- the evaluated checkpoint checksum;
- the exact test manifest;
- threshold and class-index conventions;
- per-image and aggregate metrics;
- the aggregation method and confidence interval procedure;
- raw predictions or a reproducible way to regenerate them; and
- representative failures selected by a stated rule.

For binary vessel segmentation, Dice/IoU should be accompanied by sensitivity,
specificity, and a precision-recall metric because background dominates the image.
For RAVIR, report each semantic class and a clearly defined macro average. Treat
padding and field-of-view masks explicitly.

## Why original benchmark results are absent

The repository does not contain all of the following for the exploratory runs:

- immutable dataset and split manifests;
- complete environment and hardware records;
- trained checkpoint provenance;
- held-out prediction artifacts;
- per-image metric outputs; and
- repeated seeds or uncertainty estimates.

Without that chain of evidence, publishing historical loss values as benchmark
results would be misleading. Superseded exploratory notebooks and prototypes were
removed from the maintained tree and remain recoverable from Git history for
provenance work only.

## Artifact checklist for any extension

Before making a new quantitative claim, include:

- [ ] a versioned config and Git commit;
- [ ] data-release metadata and checksums;
- [ ] image/patient-level split manifests;
- [ ] seeds and environment capture;
- [ ] checkpoint and prediction checksums;
- [ ] machine-readable per-image metrics;
- [ ] baseline and ablation definitions;
- [ ] uncertainty estimates and failure cases; and
- [ ] a statement distinguishing exploratory, validation, and test analyses.

Contribution requirements are described in [`CONTRIBUTING.md`](../CONTRIBUTING.md).

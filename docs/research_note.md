# Research note

## Scope and status

This repository is the public archive of a completed exploratory project on retinal
vessel segmentation with implicit neural representations (INRs). Giovanni Filomeno
implemented and explored the approach in research collaboration with researchers at
the Medical University of Vienna. The repository is independently maintained and is
not an official publication or endorsed institutional software.

The project is closed to new training runs. The public-hardening work performed after
the collaboration improves software contracts and documentation; it does not alter
the evidentiary status of the original experiments.

## Research question

The study asked whether a compact coordinate-based network could serve as a useful
segmentation baseline. For each pixel, the hypothesis class maps normalized position
and grayscale intensity to class logits:

$$
f_\theta(x, y, I) \rightarrow z.
$$

The approach was motivated by two properties of coordinate networks:

1. they expose a continuous query interface rather than a fixed output raster; and
2. Fourier features and sinusoidal activations can represent high-frequency signals.

Those properties motivate an experiment; they do not by themselves prove accurate
segmentation, cross-resolution generalization, or clinical utility.

## Implemented baseline

The maintained implementation contains:

- normalization of image intensities and 2D coordinates;
- Fourier encoding of spatial coordinates;
- an MLP with SIREN-style sinusoidal layers and initialization;
- one-logit binary and raw-logit multiclass output contracts;
- focal/Dice-style objectives for class-imbalanced segmentation;
- patch and dataset utilities; and
- CPU-only synthetic checks for shape, numerical, and gradient behavior.

The exploratory work investigated FIVES for binary vessel/background labels and
RAVIR for multiclass artery/vein/background labels. Superseded notebooks and
prototypes were removed from the maintained tree; they are recoverable through Git
history but are not canonical package or evidence artifacts.

## What can be concluded

The public archive establishes that the model family and data pipeline can be
implemented as inspectable PyTorch components with explicit tensor contracts. The
synthetic suite can check those software properties without private data.

The archive does **not** establish comparative segmentation performance. No result
table is published because the retained artifacts are insufficient for a defensible
benchmark: they do not include a complete immutable environment, original split
manifests, checkpoints, run logs, multiple seeds, and held-out prediction/metric
artifacts. Training losses found in superseded exploratory notebooks must not be
interpreted as Dice scores, test performance, or model rankings.

## Scientific limitations

### Pointwise conditioning

The model receives one pixel's coordinates and intensity at a time. Unlike an image
encoder, it has no explicit view of local texture, branching structure, or vessel
continuity. Coordinates may encode dataset- or crop-specific priors, but do not
replace visual context.

### Patch coordinates

When coordinates restart within each crop, identical coordinates can refer to many
anatomically unrelated locations. A claim about a continuous full-image field would
require globally defined coordinates or an image-conditioned latent representation.

### Input modality

The maintained FIVES loader converts color fundus photographs to grayscale. This
reduces the pointwise input to one intensity channel, but the original work did not
preserve an RGB-versus-grayscale ablation. Any effect on vessel contrast or
generalization is therefore unknown.

### Resolution

A coordinate model can be queried on a denser or sparser grid. Whether its predictions
remain calibrated and anatomically consistent across resolutions is an empirical
question. This project did not preserve a controlled cross-resolution evaluation, so
"resolution-independent segmentation" is not claimed.

### Evidence and generalization

The archive contains no validated test-set metrics, statistical uncertainty,
cross-dataset evaluation, clinical reader study, or robustness analysis. FIVES and
RAVIR also represent different modalities and labeling tasks; they should not be
treated as interchangeable evidence.

## A defensible future study design

Although no additional training is planned for this archive, a future replication
could test the hypothesis with the following preregistered protocol:

1. Create patient/image-level train, validation, and test manifests before patch
   extraction; publish identifiers or cryptographic hashes where licenses permit.
2. Compare the pointwise INR with at least a parameter-matched ReLU MLP, an INR
   without Fourier features, and a standard segmentation baseline such as U-Net.
3. Select checkpoints on validation metrics only and keep the test set sealed.
4. Run several fixed seeds and report per-image Dice, IoU, sensitivity, specificity,
   and precision-recall metrics; report macro per-class metrics for RAVIR.
5. Publish confidence intervals, failure cases, parameter count, peak memory, and
   latency with the exact evaluation code.
6. Test cross-resolution behavior by defining the resampling operators and target
   grids in advance, including a conventional interpolation baseline.
7. Preserve configs, dependency lock data, split manifests, checkpoints, predictions,
   and machine-readable metrics for every reported run.

An equally valuable outcome would be a carefully measured negative result showing
where pointwise conditioning fails relative to neighbourhood-aware encoders.

## Responsible-use statement

This code is a research and educational artifact. It is not a medical device, has not
been clinically validated, and must not be used to diagnose, screen, monitor, or treat
patients. Dataset access and use remain subject to the terms set by the FIVES and
RAVIR maintainers.

## Further reading

See the [bibliography](../references/README.md), [reproducibility guide](reproducibility.md),
and [third-party notices](../THIRD_PARTY_NOTICES.md).

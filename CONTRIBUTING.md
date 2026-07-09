# Contributing

Thank you for helping improve this research archive. The original training project is
complete; current maintenance focuses on correctness, clarity, reproducibility, and
safe reuse.

## Suitable contributions

- bug fixes with a focused regression test;
- stronger input, label, and tensor-shape validation;
- CPU-friendly synthetic tests;
- documentation, type annotations, and accessibility improvements;
- dependency or CI maintenance; and
- clearly scoped research extensions that do not overwrite the historical record.

Large new training campaigns, uploaded datasets, and unsupported performance claims
are outside routine maintenance scope. Discuss a substantial change in an issue
before investing significant work.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

python -m retina_inr.smoke
pytest
```

Run the configured formatter and linter before opening a pull request:

```bash
ruff format --check .
ruff check .
```

Keep tests deterministic and small enough for CPU CI.

## Pull-request expectations

A pull request should:

1. explain the user-visible problem and why the proposed behavior is correct;
2. keep the change focused and preserve backward compatibility where practical;
3. include tests for new or corrected behavior;
4. update public documentation when an interface changes;
5. avoid committing generated caches, checkpoints, run logs, or medical data; and
6. disclose limitations and any behavior that remains unverified.

Use raw logits at model boundaries unless an API is explicitly an inference-probability
helper. Validate image/mask pairing by identifier, reject unknown mask values, and use
nearest-neighbour interpolation for categorical masks.

## Requirements for quantitative claims

Do not add a result table based on training loss, a single run, or manually copied
notebook output. A new scientific claim must include, as applicable:

- a preregistered or clearly documented protocol;
- data-release metadata and image/patient-level split manifests;
- exact configs, seeds, environment, and source revision;
- checkpoint and prediction provenance;
- machine-readable per-image metrics and uncertainty estimates;
- appropriate baselines and ablations; and
- representative failure cases selected by a stated method.

See [`docs/reproducibility.md`](docs/reproducibility.md) for the complete artifact
checklist. Label exploratory observations as exploratory.

## Data, privacy, and responsible use

Never include retinal images, annotations, patient information, or dataset archives
in a contribution unless redistribution is unambiguously permitted and explicitly
approved by the repository owner. Remove local absolute paths and credentials from
notebooks and logs.

This project is not clinically validated. Contributions must not present it as a
medical device or recommend its use in patient care.

## Licensing

By contributing, you agree that your first-party contribution may be distributed
under the repository's [MIT License](LICENSE). Identify any third-party code or assets
before submission and include their provenance and compatible license; do not copy
material merely because it is publicly accessible.

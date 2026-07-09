# Data setup

No medical images, masks, derived patches, or patient-level metadata are distributed
with this repository. Download each dataset from its maintainer, review the applicable
terms, and keep all data under this ignored directory.

## FIVES

- **Dataset record:** [FIVES on Figshare](https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169)
- **Dataset paper:** [Jin et al., *Scientific Data* (2022)](https://doi.org/10.1038/s41597-022-01564-3)
- **Task used here:** binary vessel/background segmentation of color fundus images.

After downloading and extracting the archive, arrange or link the relevant directories
as follows:

```text
data/FIVES/
├── train/
│   ├── Original/
│   └── Ground truth/
└── test/
    ├── Original/
    └── Ground truth/
```

Some releases extract into an additional directory named
`FIVES A Fundus Image Dataset for AI-based Vessel Segmentation`; it is fine to keep
that wrapper if you pass the resulting paths explicitly. Do not silently mix release
layouts.

The maintained binary pipeline expects image and mask files to share a unique stem.
Masks must be categorical background/vessel masks. Verify their actual values before
mapping to `{0, 1}`; do not infer pairings from independent sort order.

## RAVIR

- **Dataset portal:** [RAVIR on Grand Challenge](https://ravir.grand-challenge.org/)
- **Dataset paper:** [Hatamizadeh et al., *IEEE Journal of Biomedical and Health Informatics* (2022)](https://doi.org/10.1109/JBHI.2022.3163352)
- **Task used here:** background/vein/artery segmentation of infrared reflectance
  images.

Expected training layout:

```text
data/RAVIR/
├── train/
│   ├── training_images/
│   └── training_masks/
└── test/                         # optional; contents depend on the release
```

The historical pipeline used raw grayscale mask values `0`, `128`, and `255`, mapped
to background `0`, vein `1`, and artery `2`, respectively. Treat that as a repository
convention to verify against the dataset documentation for the release you obtain.
Reject unknown label values instead of coercing them to background.

## Integrity and pairing checks

Before training or evaluation:

1. record the download URL, retrieval date, release identifier, and archive checksum;
2. enumerate image and mask identifiers and fail on missing, duplicate, or extra pairs;
3. verify image/mask dimensions before resizing or patch extraction;
4. inspect the complete set of mask values;
5. split full images (and patients where identifiers allow) before creating patches;
6. apply geometric transforms jointly to images and masks, but photometric transforms
   only to images; and
7. use nearest-neighbour interpolation for categorical masks.

The original study's immutable split and patch manifests were not retained. Reusing a
folder name from a superseded notebook recovered through Git history therefore does
not reproduce a historical run. See
[`docs/reproducibility.md`](../docs/reproducibility.md) for the manifest fields needed
by a future extension.

## Version-control policy

`.gitignore` excludes everything under `data/` except this file. Do not force-add
dataset archives, extracted images, masks, or derived patches. Dataset licenses and
data-use conditions are independent of this repository's MIT license.

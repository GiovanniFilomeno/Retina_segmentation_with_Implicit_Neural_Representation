# Third-party notices

The [MIT License](LICENSE) applies to first-party source code in this repository. It
does not replace the licenses, copyright, attribution requirements, data-use terms,
or access conditions of third-party material.

## LIIF reference implementation

Earlier Git revisions contained a historical snapshot of the official implementation
associated with:

> Yinbo Chen, Sifei Liu, and Xiaolong Wang. *Learning Continuous Image
> Representation with Local Implicit Image Function.* CVPR 2021.

- Upstream project: <https://github.com/yinboc/liif>
- Upstream paper: <https://arxiv.org/abs/2012.09161>
- License: BSD 3-Clause
- Copyright: Copyright (c) 2020, Yinbo Chen

The snapshot is not present in the maintained tree. Its exact upstream revision was
not recorded in the original archive; anyone inspecting or redistributing an earlier
revision must retain its bundled BSD 3-Clause license and notices. Its historical
presence does not imply that the LIIF authors endorse this project.

## Papers and research documents

Binary copies of scholarly works were removed from the maintained tree. Earlier Git
revisions may contain third-party papers; copyright remains with their respective
authors and/or publishers and those files are not relicensed under MIT. Prefer the
publisher and author links in [`references/README.md`](references/README.md) when
accessing, sharing, or citing these works.

## FIVES and RAVIR data

FIVES and RAVIR images, annotations, and metadata are **not included**. Users must
obtain them from the official sources and comply with the terms supplied by their
maintainers:

- FIVES: <https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169>
- RAVIR: <https://ravir.grand-challenge.org/>

The repository's MIT license grants no rights to those datasets.

## Runtime and development dependencies

Python packages installed through `pyproject.toml` or environment files remain under
their own licenses. Dependency names in this repository are compatibility metadata,
not a redistribution or endorsement statement. Consult each installed distribution
for its license and notices before redistribution.

If you believe an attribution or third-party artifact needs correction, open an issue
with the relevant path and authoritative provenance information.

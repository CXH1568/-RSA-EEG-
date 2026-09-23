# JCT-RSA

## Overview

Jointly Constrained Temporal RSA framework: EEG preprocessing, ridge mapping into visual embedding spaces, concept-level RDM comparison, temporal analysis, visual controls, and visualization code.

This is a **code-only release of the archived ICASSP submission workflows**, not a complete experiment reproduction package. It preserves ten scientific scripts behind explicit run functions. It does not include later 10-split, Joint-C7, retrieval, or layer-commonality results. Entry names do not identify the later provenance-clean Primary protocol.

## Installation

Use Python 3.10 and an isolated environment:

```bash
conda env create -f environment/environment.yml
conda activate jct-rsa
python -m pip install -r environment/requirements.txt
```

Install a mutually compatible PyTorch/torchvision build for your hardware before the requirements if necessary. CLIP is the OpenAI implementation, not an unrelated package with the same import name. Installing requirements may download software; running an analysis may download the original pretrained CLIP/ResNet weights through their libraries. No weights are distributed here.

The dependency declaration is not a historical lock or a guarantee of clean-install compatibility on every platform. The full research environment has not been validated for this code-only release; scientific-dependency import is **NOT_VERIFIED** and is not a packaging acceptance condition. No further environment-level dependency validation is performed. See [docs/VALIDATION.md](docs/VALIDATION.md) for the limited checks and historical probe record.

## Dataset

This repository contains **no EEG or THINGS-EEG data**, participant files, epochs, feature caches, experiment results, model checkpoints, logs, manuscript PDFs, or final figures. Download the required THINGS-EEG dataset yourself under the data provider's terms. You also need its RSVP event metadata, corresponding THINGS concept images and lawful model access; EEG alone is insufficient.

Edit `configs/paths.ini`; defaults are `./data`, `./results`, and `./configs`. Paths resolve relative to this repository, not to the original development machine.

Expected input layout (not included):

```text
data/
  sub-01/eeg/sub-01_task-rsvp_eeg.vhdr
  sub-01/eeg/sub-01_task-rsvp_events.tsv
  ... BrainVision EEG and marker files named by each header ...
  ... sub-02 through sub-50, including sub-06 ...
  images/images_THINGS/object_images/<concept>/<image>.jpg
```

Required event columns: `istarget`, `onset`, `objectnumber`, `object`. The loader preserves the archived meaning of `onset` as event sample indices. Do not substitute a differently formatted THINGS dataset or neural-RDM-only export based solely on its name.

Missing required files produce an explicit failure:

```text
Dataset not found.
Please download the required dataset and configure DATA_ROOT.
```

Missing images never produce fabricated embeddings; participant failures stop the workflow rather than reduce the cohort. No fallback to private local data is provided.

## Usage

From the repository root, inspect help first:

```bash
python scripts/run_primary.py --help
python scripts/run_temporal.py --help
python scripts/run_controls.py --help
python scripts/generate_figures.py --help
```

After independently preparing the inputs, the real analysis commands are:

```bash
python scripts/run_primary.py
python scripts/run_temporal.py
python scripts/run_controls.py
python scripts/generate_figures.py --input ./results --output ./results/figures/specificity.pdf
```

The first three commands perform real experiments when explicitly run with valid data; they are **not** smoke tests and can be expensive. They were not executed for this release. All support `--config`, `--data-root`, `--images-root`, and `--output-root`. If moving EEG and images, configure both roots.

| Entry | Actual archived workflow |
| --- | --- |
| run_primary | ridge_clip_specificity_v2: four targets, per-window fitting, single-window held-out evaluation |
| run_temporal | heldout_temporal_fixed_model: fixed mapping trained on training concepts, evaluated over time |
| run_controls | partial_rsa_variance_partition: Pixel-PCA partial Spearman controls |
| generate_figures | Visualization of your specificity temporal/held-out summaries; no model fitting or group-statistic recomputation |

The renderer requires your `clip_specificity_v2_temporal.csv` and `clip_specificity_v2_heldout.csv` in the input directory. These are runtime inputs, not distributed files. It refuses to overwrite an existing figure. It is not an exact generator of the final manuscript artwork.

The other seven workflows remain accessible as `src.workflows.<original_filename>.run(settings)`, with settings from `src.runtime.load_settings()`. See [docs/CODE_ORIGINS.md](docs/CODE_ORIGINS.md) for the source mapping. Scientific constants remain in their respective workflows to avoid changing historical protocols.

## Citation

This repository provides method implementations; it does not contain the paper's experimental data or results. Cite the associated manuscript and the original dataset/model publications as appropriate. This release does not establish an acceptance status or DOI and makes no claim that historical results were reverified.

## Limitations and license

Only syntax, import, CLI help, path/file and sensitive-information checks were performed. No scientific workflow or renderer was run, and numerical equivalence was not tested.

Here, import checks refer to this repository's import-safe modules, not the full scientific dependency stack. **EXPERIMENTS = NOT_RUN; NUMERICAL_RESULTS = NOT_VERIFIED; CODE_PACKAGE_ONLY = YES.**

The archived loader uses prefix-truncated event labels after epoch selection. Its event alignment is not certified here. The archived full-cohort BH routine also has a rank-weight ordering concern and is preserved, not scientifically repaired. Do not interpret import success as validation of event alignment, FDR values or the legacy interpretation strings written by the scripts. These issues require separate scientific review before relying on new inferential claims.

MIT License, with the original copyright retained. Third-party packages, datasets, images and pretrained weights retain their own licenses. Nothing in this repository relicenses those resources.

# whole_embryo_analysis
<<<<<<< HEAD

Analysis and visualization pipeline for whole-embryo light sheet microscopy data. Performs nuclear shape outlier detection, PCA-based shape space analysis on spherical harmonic coefficients, and somite-level feature extraction.

## Pipeline Overview

The workflow runs three sequential stages via Snakemake:

1. **KDE Outlier Detection** — Pairwise kernel density estimation on nuclear shape metrics (volume, axis lengths) to identify and remove morphological outliers.
2. **Shapespace PCA** — Principal component analysis on spherical harmonic coefficients, producing per-nucleus PC scores, reconstructed shape mode meshes (VTK), and diagnostic figures.
3. **Somite Feature Extraction** — Maps nuclei to somites, computes biological coordinate frames (anatomical/Frenet/hybrid), depth, and orientation features per nucleus.

## Installation

```bash
conda activate yourenv
pip install -e .
```

## Configuration

All paths and parameters live in `config/config.yaml`. Key settings:

- `feature_csv` — Path to the nuclear feature CSV (with `shcoeffs_*` columns)
- `somite_lut` — Path to the merged somite lookup table (all embryos)
- `embryo_id` — Which embryo to process (filters the LUT)
- `output_dir` — Root directory for all pipeline outputs
- `kde`, `shapespace`, `somite` — Per-stage parameters and Dask cluster settings

## Running the Pipeline

Dry run (preview what will execute):

```bash
snakemake -np --configfile config/config.yaml
```

Full pipeline:

```bash
snakemake -p --configfile config/config.yaml --cores 1
```

Run a subset of the pipeline:

```bash
snakemake -p --until shapespace_pca --configfile config/config.yaml --cores 1
```

Submit as an SGE job (recommended for long runs):

```bash
qsub submit_snakemake.sh
```

## Project Structure

```
whole_embryo_analysis/
├── Snakefile                   # Workflow definition (3 rules)
├── config/config.yaml          # All paths and parameters
├── submit_snakemake.sh         # SGE job wrapper for Snakemake
├── pyproject.toml              # Package metadata and dependencies
├── src/whole_embryo_analysis/  # Installable Python package
│   ├── find_outliers.py        # Stage 1: KDE outlier detection
│   ├── shapespace/             # Stage 2: PCA on SH coefficients
│   ├── somite_analysis/        # Stage 3: Somite feature extraction
│   └── utils/                  # Shared utilities (Dask, I/O, viz)
├── notebooks/                  # Exploratory and visualization notebooks
├── figures/                    # Generated figures and meshes
└── logs/                       # Snakemake and per-rule logs
```
=======
This repository will contain all notebooks and analytical steps resulting from the lightsheet analysis pipeline: https://github.com/beliveau-lab/lightsheet-analysis-pipeline
>>>>>>> 2fc83721d629d2f2ac50fd832950eb887d8f2f43

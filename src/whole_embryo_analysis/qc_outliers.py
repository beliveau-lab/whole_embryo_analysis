#!/usr/bin/env python3
"""Visual QC for outlier detection: crop N random flagged nuclei and render
max-intensity projections along three axes with the microscopy image masked
to the nucleus.

Usage (standalone):
    python -m whole_embryo_analysis.qc_outliers \
        --outlier_csv  /path/to/outliers.csv \
        --feature_path /path/to/nuclear_features.parquet \
        --nuclear_mask /path/to/dataset_fused_masks.zarr \
        --image_path   /path/to/dataset_fused.n5 \
        --channel 2 \
        --output_dir   /path/to/qc_figures \
        --n 16

Produces one figure per nucleus (3-axis MIP) plus a combined grid figure
for quick visual screening.
"""
import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import zarr
import dask.array as da

from whole_embryo_analysis.utils.io import scan_features
from whole_embryo_analysis.utils.madi_utils import load_n5_zarr_array

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Core helpers
# ═════════════════════════════════════════════════════════════════════════════

def crop_nucleus(nuclear_id, centroid_z, centroid_y, centroid_x,
                 mask_da, image_da, crop_size=256):
    """Centroid-centered crop, masked to the nucleus.

    Matches the notebook pattern: fixed crop window around centroid,
    binary mask from the segmentation, np.where to zero background.

    Parameters
    ----------
    nuclear_id : int
        Label value in the segmentation mask.
    centroid_z, centroid_y, centroid_x : float
        Centroid coordinates of the nucleus.
    mask_da : dask.array.Array
        Nuclear segmentation mask (ZYX).
    image_da : zarr.Array
        Single-channel microscopy image (ZYX).
    crop_size : int
        Side length of the crop cube in voxels.

    Returns
    -------
    img_crop : np.ndarray (3-D)
        Masked microscopy crop (background zeroed).
    """
    z_dim, y_dim, x_dim = image_da.shape
    z, y, x = int(centroid_z), int(centroid_y), int(centroid_x)
    half = crop_size // 2

    z1, z2 = max(0, z - half), min(z_dim, z + half)
    y1, y2 = max(0, y - half), min(y_dim, y + half)
    x1, x2 = max(0, x - half), min(x_dim, x + half)

    binary_mask = np.asarray(mask_da[z1:z2, y1:y2, x1:x2].compute()) == nuclear_id
    img_crop = np.asarray(image_da[z1:z2, y1:y2, x1:x2])
    img_crop = np.where(binary_mask, img_crop, 0)

    return img_crop


def plot_nucleus_mips(img_crop, nuclear_id):
    """Create a 1x3 figure: MIP along Z, Y, X axes.

    Parameters
    ----------
    img_crop : np.ndarray (3-D)
        Masked microscopy crop (ZYX).
    nuclear_id : int
    outlier_info : str
        Extra text for the suptitle (e.g. outlier reason).

    Returns
    -------
    matplotlib.figure.Figure
    """
    axis_labels = ["Z projection"]

    fig, axes = plt.subplots(1, 1, figsize=(6, 2))
    axes.imshow(np.max(img_crop, axis=0))
    axes.set_title(axis_labels[0], fontsize=9)
    axes.axis("off")

    suptitle = f"Nuclear_ID {nuclear_id}"
    fig.suptitle(suptitle, fontsize=10, )
    fig.tight_layout()
    return fig


def plot_grid(grid_data, n_cols=4, title="QC: Random Sample"):
    """Combine per-nucleus MIPs into a single grid image.

    Parameters
    ----------
    grid_data : list of (nuclear_id, img_crop)
    n_cols : int
    title : str
        Suptitle for the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(grid_data)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                            figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for idx, (nid, img_crop) in enumerate(grid_data):
        row = idx // n_cols
        col = idx % n_cols

        ax = axes[row, col]
        ax.imshow(np.max(img_crop, axis=0), cmap="gray")
        ax.set_title(f"ID {nid}", fontsize=7)
        ax.axis("off")

    # Hide unused axes
    for idx in range(n, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def _crop_and_save(sampled_ids, feature_df, mask_da, image_da,
                   crop_size, out_dir, prefix):
    """Crop each sampled nucleus, save per-nucleus PNG, return grid_data."""
    grid_data = []
    for nid in sampled_ids:
        nid_int = int(nid)

        centroid_row = feature_df.filter(pl.col("Nuclear_ID") == nid_int)
        if len(centroid_row) == 0:
            logger.warning("Nuclear_ID %d not found in feature table, skipping", nid_int)
            continue

        cz = centroid_row["Centroid_Z"][0]
        cy = centroid_row["Centroid_Y"][0]
        cx = centroid_row["Centroid_X"][0]

        img_crop = crop_nucleus(
            nid_int, cz, cy, cx, mask_da, image_da, crop_size=crop_size,
        )
        if img_crop.max() == 0:
            logger.warning("Nuclear_ID %d: empty crop, skipping", nid_int)
            continue

        fig = plot_nucleus_mips(img_crop, nid_int)
        fig.savefig(out_dir / f"{prefix}_{nid_int}.png", dpi=300,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)
        grid_data.append((nid_int, img_crop))
    return grid_data

# ═════════════════════════════════════════════════════════════════════════════
# Main workflow
# ═════════════════════════════════════════════════════════════════════════════

def qc_outliers(
    outlier_csv: str,
    feature_path: str,
    nuclear_mask_path: str,
    image_path: str,
    channel: int = 2,
    mask_subpath: str | None = None,
    output_dir: str = "qc_outliers",
    n: int = 16,
    crop_size: int = 64,
    seed: int | None = None,
):
    """Sample N random outlier nuclei, crop, and generate MIP figures.

    Parameters
    ----------
    outlier_csv : str
        Path to outliers.csv from find_outliers (must contain Nuclear_ID).
    feature_path : str
        Path to feature table (CSV or Parquet) with Centroid_Z/Y/X columns.
    nuclear_mask_path : str
        Path to segmentation mask (.zarr or .n5).
    image_path : str
        Path to microscopy image (.zarr or .n5) with channel subdirectories.
    channel : int
        Image channel to visualize (default 2).
    mask_subpath : str or None
        Subpath within mask zarr/n5.
    output_dir : str
        Directory for output figures.
    n : int
        Number of random outliers to visualize.
    crop_size : int
        Side length of crop cube in voxels (default 256).
    seed : int
        Random seed for reproducibility.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load outlier IDs
    outlier_df = scan_features(outlier_csv).collect()
    logger.info("Loaded %d outliers from %s", len(outlier_df), outlier_csv)

    # Load centroids from feature table
    feature_df = (
        scan_features(feature_path)
        .select(["Nuclear_ID", "Centroid_Z", "Centroid_Y", "Centroid_X"])
        .collect()
    )
    logger.info("Loaded %d features from %s", len(feature_df), feature_path)

    # Sample N random outlier IDs + N matched-size non-outlier IDs
    rng = np.random.default_rng(seed)
    outlier_ids = outlier_df["Nuclear_ID"].to_numpy()
    all_feature_ids = feature_df["Nuclear_ID"].to_numpy()
    outlier_set = set(outlier_ids.tolist())
    normal_ids = np.array(
        [i for i in all_feature_ids if int(i) not in outlier_set],
        dtype=all_feature_ids.dtype,
    )
    logger.info("%d outliers, %d kept nuclei available for sampling",
                len(outlier_ids), len(normal_ids))

    n_out = min(n, len(outlier_ids))
    n_norm = min(n, len(normal_ids))
    sampled_outliers = rng.choice(outlier_ids, size=n_out, replace=False)
    sampled_normals = rng.choice(normal_ids, size=n_norm, replace=False)
    logger.info("Sampled %d outliers + %d kept nuclei", n_out, n_norm)

    # Load mask via load_n5_zarr_array (works for both .zarr and .n5)
    mask_da = load_n5_zarr_array(nuclear_mask_path, n5_subpath=mask_subpath)
    logger.info("Mask shape: %s", mask_da.shape)

    # Load image channel — open root store, then access ch/s0 via dict lookup
    image_store = zarr.open(image_path, mode="r")
    image_da = image_store[f"ch{channel}"]["s0"]
    logger.info("Image ch%d shape: %s", channel, image_da.shape)

    # Crop + save per-nucleus figures for both groups
    outlier_grid = _crop_and_save(
        sampled_outliers, feature_df, mask_da, image_da,
        crop_size, out, prefix="qc_outlier",
    )
    normal_grid = _crop_and_save(
        sampled_normals, feature_df, mask_da, image_da,
        crop_size, out, prefix="qc_normal",
    )

    if not outlier_grid and not normal_grid:
        logger.error("No nuclei could be cropped — check IDs and feature table")
        return

    if outlier_grid:
        fig = plot_grid(outlier_grid, n_cols=4,
                        title="Outliers: Random Sample")
        fig.savefig(out / "qc_grid_outliers.png", dpi=150,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)

    if normal_grid:
        fig = plot_grid(normal_grid, n_cols=4,
                        title="Kept nuclei: Random Sample")
        fig.savefig(out / "qc_grid_normals.png", dpi=150,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)

    logger.info("Saved %d outlier + %d normal figures to %s",
                len(outlier_grid), len(normal_grid), out)


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Visual QC: crop random outlier nuclei and render MIPs",
    )
    p.add_argument("--outlier_csv", required=True,
                   help="outliers.csv from find_outliers")
    p.add_argument("--feature_path", required=True,
                   help="Feature table (CSV or Parquet) with Centroid_Z/Y/X")
    p.add_argument("--nuclear_mask", required=True,
                   help="Segmentation mask (.zarr or .n5)")
    p.add_argument("--image_path", required=True,
                   help="Microscopy image (.zarr or .n5) with channel subdirs")
    p.add_argument("--channel", type=int, default=2,
                   help="Image channel to visualize (default 2)")
    p.add_argument("--mask_subpath", default=None,
                   help="Subpath within mask zarr/n5")
    p.add_argument("--output_dir", default="qc_outliers",
                   help="Output directory for figures")
    p.add_argument("--n", type=int, default=16,
                   help="Number of random outliers to visualize (default 16)")
    p.add_argument("--crop_size", type=int, default=256,
                   help="Crop cube side length in voxels (default 256)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default 42)")
    return p.parse_args()


def main():
    args = parse_args()
    qc_outliers(
        outlier_csv=args.outlier_csv,
        feature_path=args.feature_path,
        nuclear_mask_path=args.nuclear_mask,
        image_path=args.image_path,
        channel=args.channel,
        mask_subpath=args.mask_subpath,
        output_dir=args.output_dir,
        n=args.n,
        crop_size=args.crop_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

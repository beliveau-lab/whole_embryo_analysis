"""
Phase 3: Hierarchical clustering of somite nuclei.

Reads the enriched somite features CSV produced by Phase 2, derives
``is_interior`` and ``ripleys_k_density``, clusters with Ward's linkage,
and saves a dendrogram figure plus an L-R enrichment line plot.

Usage:
    python -m whole_embryo_analysis.somite_analysis.somite_cluster \
        --somite_csv ... --save_dir ... --embryo_id 306
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, set_link_color_palette
from sklearn.preprocessing import StandardScaler

import tifffile

from whole_embryo_analysis.somite_analysis.plots import (
    add_binary_depth,
    add_ripleys_k_density,
    get_cluster_colors,
    load_somite_roi,
    plot_somite_mip,
    print_feature_ranges,
)
from whole_embryo_analysis.utils.io import load_n5_zarr_array
from whole_embryo_analysis.utils.viz import save_vectorized

logger = logging.getLogger(__name__)

FEATURE_COLS = ["PC1", "PC2", "PC3", "is_interior", "ripleys_k_density", "align_AP", "align_LR", "align_DV"]


# ── helpers ──────────────────────────────────────────────────────────────────

def _resolve_side_col(df: pd.DataFrame) -> str:
    """Return the column name that holds the L/R side label."""
    for candidate in ("Somite_Side", "Side"):
        if candidate in df.columns:
            return candidate
    raise KeyError("No side column found (expected 'Side' or 'Somite_Side')")


def _resolve_stage_col(df: pd.DataFrame) -> str:
    """Return the column name that holds the somite stage."""
    for candidate in ("somite_stage", "Somite_Stage"):
        if candidate in df.columns:
            return candidate
    raise KeyError("No stage column found (expected 'Somite_Stage' or 'somite_stage')")


def _normalise_side(series: pd.Series) -> pd.Series:
    """Map side values to 'Left' / 'Right' regardless of case."""
    return series.str.strip().str.capitalize()


# ── clustering ───────────────────────────────────────────────────────────────

def cluster_ward(df, feature_cols, n_clusters=2):
    """Ward's hierarchical clustering. Returns (df_with_labels, Z linkage)."""
    X = df[feature_cols].values
    X = StandardScaler().fit_transform(X)
    X = np.nan_to_num(X, nan=0.0)

    Z = linkage(X, method="ward")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    df = df.copy()
    df["cluster_hier"] = labels
    return df, Z


def plot_dendrogram(Z, n_clusters, save_dir, file_name="dendrogram"):
    """Save a dendrogram coloured by cluster."""
    colors = get_cluster_colors()
    set_link_color_palette([colors[i] for i in range(1, n_clusters + 1)])

    fig, ax = plt.subplots(figsize=(10, 4))
    dendrogram(
        Z,
        color_threshold=_ward_threshold(Z, n_clusters),
        above_threshold_color="gray",
        no_labels=True,
        ax=ax,
    )
    ax.legend(
        handles=[Patch(facecolor=colors[c], label=f"Cluster {c}")
                 for c in range(1, n_clusters + 1)],
        loc="upper right", frameon=False, labelcolor="white",
    )
    ax.set_ylabel("Distance")
    ax.set_title("Ward Dendrogram")
    save_vectorized(save_dir, fig, file_name, transparent=True)
    plt.close(fig)
    logger.info(f"Saved dendrogram to {save_dir}/{file_name}.svg")

# def plot_dendrogram(Z, labels, n_clusters, save_dir, file_name="dendrogram"):
#     """Save a dendrogram with links coloured by cluster, plus a legend."""
#     colors = get_cluster_colors()
#     n = len(labels)

#     # Cluster per node: leaves from fcluster, merges inherit only if children agree
#     node = {i: int(labels[i]) for i in range(n)}
#     for k, (a, b, *_) in enumerate(Z):
#         node[n + k] = node[a] if node[a] == node[b] else -1

#     fig, ax = plt.subplots(figsize=(10, 4))
#     dendrogram(
#         Z,
#         link_color_func=lambda i: colors.get(node[i], "gray"),
#         no_labels=True,
#         ax=ax,
#     )
#     ax.legend(
#         handles=[Patch(facecolor=colors[c], label=f"Cluster {c}")
#                  for c in range(1, n_clusters + 1)],
#         loc="upper right", frameon=False, labelcolor="white",
#     )
#     ax.set_ylabel("Ward merge distance")
#     ax.set_title("Ward Dendrogram")
#     save_vectorized(save_dir, fig, file_name, transparent=True)
#     plt.close(fig)
#     logger.info(f"Saved dendrogram to {save_dir}/{file_name}.svg")

def _ward_threshold(Z, n_clusters):
    """Find the merge distance that yields exactly *n_clusters*."""
    if n_clusters <= 1:
        return Z[-1, 2] + 1
    return (Z[-(n_clusters - 1), 2] + Z[-n_clusters, 2]) / 2


# ── L-R enrichment plot ─────────────────────────────────────────────────────

def plot_lr_enrichment(df, save_dir, file_name="lr_enrichment",
                       side_col="Side", stage_col="Somite_Stage"):
    """
    For each cluster, count nuclei per (stage, side), compute Left − Right,
    and plot as a line chart.
    """
    side = _normalise_side(df[side_col])

    pivot_L = (
        df.assign(**{side_col: side})
        .loc[side == "Left"]
        .groupby([stage_col, "cluster_hier"])
        .size()
        .unstack(fill_value=0)
    )
    pivot_R = (
        df.assign(**{side_col: side})
        .loc[side == "Right"]
        .groupby([stage_col, "cluster_hier"])
        .size()
        .unstack(fill_value=0)
    )

    pivot_L, pivot_R = pivot_L.align(pivot_R, fill_value=0)
    diff = pivot_L - pivot_R

    colors = get_cluster_colors()
    clusters = sorted(diff.columns.tolist())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    markers = ["o", "s", "^", "D", "v", "p", "*", "X"]
    for i, c in enumerate(clusters):
        ax.plot(
            diff.index, diff[c],
            marker=markers[i % len(markers)],
            label=f"Cluster {c}",
            color=colors.get(c, "black"),
        )

    legend_handles = [
        Patch(facecolor=colors.get(c, "black"), label=f"Cluster {c}")
        for c in clusters
    ]
    ax.legend(
        handles=legend_handles, title="",
        bbox_to_anchor=(1.05, 1), frameon=False, labelcolor="white",
    )
    ax.set_xticks(diff.index)
    ax.set_xlabel("Somite stage")
    ax.set_ylabel("L \u2212 R count")
    ax.set_title("L \u2212 R enrichment across stages")
    plt.tight_layout()

    save_vectorized(save_dir, fig, file_name, transparent=True)
    plt.close(fig)
    logger.info(f"Saved L-R enrichment plot to {save_dir}/{file_name}.svg")


# ── MIP visualization ────────────────────────────────────────────────────────

def show_mips(df, nuc_mask, somite_mask, save_dir,
              side_col="Side", stage_col="Somite_Stage",
              selected_stages=None, z_factor=0.1):
    """
    For each selected somite stage, render left/right cluster MIPs
    side-by-side and save the figure.
    """
    df = df.copy()
    df[side_col] = _normalise_side(df[side_col])

    if side_col != "Somite_Side":
        df["Somite_Side"] = df[side_col]
    if stage_col != "somite_stage":
        df["somite_stage"] = df[stage_col]

    stages = sorted(df[stage_col].unique())
    if selected_stages is not None:
        stages = [s for s in stages if s in selected_stages]

    for stage in stages:
        df_stage = df[df[stage_col] == stage]
        logger.info(f"Rendering MIP for stage {stage}")

        left_df = df_stage[df_stage[side_col] == "Left"]
        right_df = df_stage[df_stage[side_col] == "Right"]

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        if not left_df.empty:
            left_result = load_somite_roi(
                nuclear_mask_da=nuc_mask, somite_mask=somite_mask,
                df_features=left_df,
            )
            plot_somite_mip(
                somite_df=left_df, result=left_result,
                color_col="cluster_hier", ax=axes[0], z_factor=z_factor,
            )
        else:
            axes[0].set_title("No left somites")
            axes[0].axis("off")

        if not right_df.empty:
            right_result = load_somite_roi(
                nuclear_mask_da=nuc_mask, somite_mask=somite_mask,
                df_features=right_df,
            )
            plot_somite_mip(
                somite_df=right_df, result=right_result,
                color_col="cluster_hier", ax=axes[1], z_factor=z_factor,
            )
        else:
            axes[1].set_title("No right somites")
            axes[1].axis("off")

        fig.suptitle(f"Stage: {stage}", color="white")
        save_vectorized(save_dir, fig, f"stage_{stage}_cluster_mip", transparent=True)
        plt.close(fig)
        logger.info(f"Saved MIP for stage {stage}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 3: hierarchical clustering of somite nuclei")

    parser.add_argument("--somite_csv", required=True,
                        help="somite_features CSV from Phase 2")
    parser.add_argument("--save_dir", required=True,
                        help="Directory for output figures and clustered CSV")
    parser.add_argument("--embryo_id", type=int, required=True)
    parser.add_argument("--n_clusters", type=int, default=2,
                        help="Number of clusters (default 2)")
    parser.add_argument("--nuclear_mask_path", required=True,
                        help="Zarr path to nuclear segmentation mask")
    parser.add_argument("--somite_mask_path", required=True,
                        help="TIFF path to somite mask")
    parser.add_argument("--mip_stage", default=None,
                        help="Somite stage to render MIP for (default 13)")
    return vars(parser.parse_args())


# ── main ─────────────────────────────────────────────────────────────────────

def run(args):
    save_dir = Path(args["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    embryo_id = args["embryo_id"]
    n_clusters = args["n_clusters"]

    logger.info(f"Loading somite features from {args['somite_csv']}")
    df = pd.read_csv(args["somite_csv"])
    logger.info(f"Loaded {len(df)} nuclei")

    side_col = _resolve_side_col(df)
    stage_col = _resolve_stage_col(df)

    df = add_binary_depth(df)
    df = add_ripleys_k_density(df)
    logger.info("Derived is_interior and ripleys_k_density")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    print_feature_ranges(df[FEATURE_COLS])

    df, Z = cluster_ward(df, FEATURE_COLS, n_clusters=n_clusters)
    logger.info(f"Clustered into {n_clusters} groups via Ward's linkage")

    fig_dir = save_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_dendrogram(Z, n_clusters, save_dir=str(fig_dir)) #(Z, n_clusters, save_dir, file_name="dendrogram"):
    plot_lr_enrichment(df, str(fig_dir), file_name="lr_enrichment",
                       side_col=side_col, stage_col=stage_col)

    out_csv = save_dir / f"somite_clustered_{embryo_id}.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved clustered CSV to {out_csv}")

    mip_stage = args.get("mip_stage", 13)
    logger.info(f"Loading masks for MIP visualization (stage {mip_stage})")
    nuc_mask = load_n5_zarr_array(args["nuclear_mask_path"])
    somite_mask = tifffile.imread(args["somite_mask_path"])
    show_mips(df, nuc_mask, somite_mask, str(fig_dir),
              side_col=side_col, stage_col=stage_col,
              selected_stages=None) # none to force rendering all


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    start_time = time.time()
    args = parse_args()
    run(args)
    logger.info(f"Phase 3 (clustering) completed in {time.time() - start_time:.1f}s")

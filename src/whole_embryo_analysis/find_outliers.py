#!/usr/bin/env python3
"""Segmentation-error detection via robust Mahalanobis distance.

Features are log-transformed and a Minimum Covariance Determinant (MCD)
estimator fits center and scatter on the least-contaminated subset, so gross
outliers don't pull the ellipsoid toward themselves. Per-nucleus squared
Mahalanobis distances are thresholded against a chi-squared quantile.
"""
import argparse
import logging
import time
from pathlib import Path

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from scipy.stats import chi2
from sklearn.covariance import MinCovDet

from whole_embryo_analysis.utils.io import scan_features
from whole_embryo_analysis.qc_outliers import qc_outliers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
sns.set_palette(cc.glasbey_dark)
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.format"] = "svg"


# ═══════════════════════════════════════════════════════════════════════════════
# Detection
# ═══════════════════════════════════════════════════════════════════════════════

def mahalanobis_outliers(df, feature_cols, alpha=0.001, support_fraction=0.9):
    """Flag nuclei whose robust Mahalanobis distance exceeds a chi-squared threshold.

    Parameters
    ----------
    df : pl.DataFrame
    feature_cols : list[str]
    alpha : float
        Significance level for the chi-squared test (e.g. 0.001 = 99.9%).
    support_fraction : float
        Fraction of points MCD uses for the covariance estimate. 0.9 tolerates
        up to .1 contamination.

    Returns
    -------
    pl.Series (bool), np.ndarray, float
        Boolean outlier mask, per-nucleus squared Mahalanobis distances, and
        the chi-squared threshold used.
    """
    data = df.select(feature_cols).to_numpy().astype(np.float64)
    data = np.log1p(np.clip(data, 0, None))

    mcd = MinCovDet(support_fraction=support_fraction, random_state=0).fit(data)
    d2 = mcd.mahalanobis(data)

    threshold = chi2.ppf(1 - alpha, df=len(feature_cols))
    mask = pl.Series("mahal_outlier", d2 > threshold)
    return mask, d2, threshold


# ═══════════════════════════════════════════════════════════════════════════════
# Detector class
# ═══════════════════════════════════════════════════════════════════════════════

class OutlierDetection:
    def __init__(self, params):
        self.figure_dir = params["figure_dir"]
        self.csv_path = params["csv_path"]
        self.outlier_path = params["outlier_path"]
        self.alpha = params.get("alpha", 0.001)

        self.normal_color = "#36454F"
        self.outlier_color = cc.glasbey_dark[0]

        self.qc_nuclear_mask = params.get("qc_nuclear_mask")
        self.qc_image_path = params.get("qc_image_path")
        self.qc_channel = params.get("qc_channel", 2)
        self.qc_n = params.get("qc_n", 16)

        self.id_col = "Nuclear_ID"
        self.feature_cols = [
            "Nuclear_Volume",
            "Major_Axis_Length",
            "Intermediate_Axis_Length",
            "Minor_Axis_Length",
        ]

        needed = [self.id_col] + self.feature_cols
        self.df = (
            scan_features(self.csv_path)
            .select(needed)
            .with_columns(pl.col(c).cast(pl.Float32) for c in self.feature_cols)
            .collect()
        )
        logger.info(f"Loaded {len(self.df)} nuclei with features: {self.feature_cols}")

    def workflow(self):
        logger.info("--- Robust Mahalanobis (MCD, alpha=%.4f) ---", self.alpha)
        mask, d2, threshold = mahalanobis_outliers(
            self.df, self.feature_cols, alpha=self.alpha,
        )
        logger.info(f"  Chi-squared threshold (df={len(self.feature_cols)}): {threshold:.2f}")
        logger.info(f"  Outliers: {int(mask.sum())} / {len(self.df)}")

        pcts = np.percentile(d2, [50, 90, 95, 99, 99.5, 99.9, 100])
        logger.info(
            "  d^2 percentiles: p50=%.1f  p90=%.1f  p95=%.1f  p99=%.1f  "
            "p99.5=%.1f  p99.9=%.1f  max=%.1f",
            *pcts,
        )

        outlier_df = self.df.filter(mask).with_columns(
            pl.Series("mahalanobis_d2", d2[mask.to_numpy()]),
        )
        outlier_df.write_csv(self.outlier_path)
        logger.info(f"Saved {len(outlier_df)} outliers to {self.outlier_path}")

        self._plot_mahalanobis_hist(d2, threshold)
        self._plot_pairplot(mask)

        if self.qc_nuclear_mask and self.qc_image_path:
            logger.info("Generating QC crops for random outliers")
            qc_outliers(
                outlier_csv=self.outlier_path,
                feature_path=self.csv_path,
                nuclear_mask_path=self.qc_nuclear_mask,
                image_path=self.qc_image_path,
                channel=self.qc_channel,
                output_dir=str(Path(self.figure_dir) / "qc_crops"),
                n=self.qc_n,
            )
        else:
            logger.info("Skipping QC crops (--qc_nuclear_mask / --qc_image_path not set)")

    # ── Plotting ──────────────────────────────────────────────────────────────

    def _save_fig(self, fig, name):
        out = Path(self.figure_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{name}.svg", format="svg",
                    bbox_inches="tight", facecolor="white")
        logger.info(f"Saved plot: {out / name}.svg")
        plt.close(fig)

    def _plot_mahalanobis_hist(self, d2, threshold):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(d2, bins=200, color=self.normal_color, alpha=0.7,
                     edgecolor="none", log=True)
        axes[0].axvline(threshold, color=self.outlier_color, ls="--", lw=2,
                        label=f"chi2 threshold = {threshold:.1f}")
        axes[0].set_xlabel("Mahalanobis d$^2$")
        axes[0].set_ylabel("Count (log scale)")
        axes[0].set_title("Full Distribution")
        axes[0].legend()

        p95 = np.percentile(d2, 95)
        tail = d2[d2 > p95]
        axes[1].hist(tail, bins=100, color=self.normal_color, alpha=0.7,
                     edgecolor="none")
        axes[1].axvline(threshold, color=self.outlier_color, ls="--", lw=2,
                        label=f"threshold = {threshold:.1f} ({(d2 > threshold).sum()} flagged)")
        axes[1].set_xlabel("Mahalanobis d$^2$")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Tail (above 95th percentile)")
        axes[1].legend(fontsize=8)

        fig.suptitle("Mahalanobis Distance Distribution", fontsize=14)
        fig.tight_layout()
        self._save_fig(fig, "mahalanobis_distribution")
        
    def _plot_pairplot(self, outlier_mask):
        df_plot = self.df.select(self.feature_cols).to_pandas()
        is_out = outlier_mask.to_numpy()
        df_plot["outlier_status"] = pd.Categorical(
            np.where(is_out, "Outlier", "Normal"),
            categories=["Normal", "Outlier"], ordered=True,
        )

        n_total, n_out_true = len(df_plot), int(is_out.sum())
        pct = 100 * n_out_true / n_total

        # Proportional subsample: preserves the true outlier fraction
        target = 20_000
        if n_total > target:
            df_plot = df_plot.sample(n=target, random_state=0)

        g = sns.pairplot(
            df_plot,
            hue="outlier_status",
            hue_order=["Normal", "Outlier"],       # normals drawn first, outliers on top
            palette={"Normal": self.normal_color, "Outlier": self.outlier_color},
            plot_kws={"alpha": 0.25, "s": 6, "edgecolor": "none", "linewidth": 0},
        )
        g.fig.suptitle(
            f"Feature Pairplot — {n_out_true:,} outliers / {n_total:,} nuclei ({pct:.1f}%)",
            y=1.02, fontsize=14,
        )
        self._save_fig(g.fig, "pairplot")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Segmentation-error detection via robust Mahalanobis distance"
    )
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Feature table (CSV or Parquet)")
    parser.add_argument("--figure_dir", type=str, required=True)
    parser.add_argument("--outlier_path", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.001,
                        help="Chi-squared significance level (default 0.001 = 99.9%%)")
    parser.add_argument("--qc_nuclear_mask", type=str, default=None,
                        help="Segmentation mask (.zarr/.n5) for QC figures.")
    parser.add_argument("--qc_image_path", type=str, default=None,
                        help="Microscopy image (.zarr/.n5) with ch<N>/s0 layout.")
    parser.add_argument("--qc_channel", type=int, default=2)
    parser.add_argument("--qc_n", type=int, default=16,
                        help="Number of random outliers to crop for QC")
    return parser.parse_args()


def main():
    args = parse_args()

    params = {
        "figure_dir": args.figure_dir,
        "csv_path": args.csv_path,
        "outlier_path": args.outlier_path,
        "alpha": args.alpha,
        "qc_nuclear_mask": args.qc_nuclear_mask,
        "qc_image_path": args.qc_image_path,
        "qc_channel": args.qc_channel,
        "qc_n": args.qc_n,
    }

    logger.info("--------------------------------")
    logger.info("Starting outlier detection (Mahalanobis-only, alpha=%.4f)", args.alpha)
    logger.info("--------------------------------")
    start_time = time.time()

    detector = OutlierDetection(params)
    detector.workflow()

    logger.info("Completed in %.2f seconds", time.time() - start_time)


if __name__ == "__main__":
    main()
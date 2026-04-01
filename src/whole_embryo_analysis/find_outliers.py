#!/usr/bin/env python3
import argparse
import logging
import time
from pathlib import Path

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.utils import resample

from dask import delayed
from whole_embryo_analysis.utils.dask_utils import (
    setup_dask_sge_cluster, shutdown_dask, add_dask_args, dask_config_from_args,
)

print("Finished imports")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Style and vector outputs
sns.set_style("whitegrid")
sns.set_palette(cc.glasbey_dark)
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.format"] = "svg"

def compute_median_density(metric_x, 
                           metric_y,
                           x_values,
                           y_values,
                           rounds,
                           n_subsample):
    """
    Compute median probabilistic density estimates for datapoints specific metric pair
    Picklable for Dask (each metric pair is computed in parallel - one core per pair)

    Params:
    ------
        metrix_x: Name of first metric
        metric_y: Name of second metric
        x_values: List of first metric values
        y_values: List of second metric values
        rounds: Number of rounds of Monte Carlo estimation
        n_subsample: Size of random sub sample per round

    Returns:
    ------
        tuple: first element is string of metric x vs y 
             second element is list of median densities for that pair
    """
    n = len(x_values)
    density_estimates = []
    for i in range(rounds):
        xS, yS = resample(
            x_values, y_values,
            replace=False,
            n_samples=min(n_subsample, n),
            random_state=i
        )
        kde = gaussian_kde(np.vstack([xS, yS]))
        dens = kde(np.vstack([x_values, y_values]))
        dens = dens / dens.sum()
        density_estimates.append(dens)
    median_densities = np.median(density_estimates, axis=0)
    return f"{metric_x}_vs_{metric_y}", median_densities


class KDEOutlierDetection:
    def __init__(self, params):
        self.figure_dir = params["figure_dir"]
        self.density_thresh = params["density_thresh"]
        self.csv_path = params["csv_path"]
        self.outlier_path = params["outlier_path"]
        self.n_subsample = params["n_subsample"]
        self.n_rounds = params["n_rounds"]

        self.df = None # df of original csv
        self.df_plot = None # df of plotting data

        self.normalize_factor = 1000 # factor to normalize data - bring all µm quantities to 10-1000 range
        self.normal_color = cc.glasbey_dark[2] # consistent color schematics
        self.outlier_color = cc.glasbey_dark[0]
        self.init_data()

    def init_data(self):
        """
        Initalize data for KDE estimation

        Returns:
        ------
            None
        """
        metrics = [
        # list of metrics for pairwise desnity estimation
            "Nuclear_ID",
            "Nuclear_Volume",
            "Major_Axis_Length",
            "Intermediate_Axis_Length",
            "Minor_Axis_Length",
        ]
        self.df = (
            pd.read_csv(
                self.csv_path,
                usecols=metrics,
                index_col=metrics[0], # index by nuclear id
                dtype={
                    metrics[1]: "float32",
                    metrics[2]: "float32",
                    metrics[3]: "float32",
                    metrics[4]: "float32",
                }
            )
        )
        logger.info(f"Using metrics: {list(self.df.columns)}. {len(self.df)} objects loaded")
        return 
    
    def permute_pairs(self):
        """
        Create permuted list of pairs to compute density of

        Returns
        -------
        pairs: list of tuples of feature pairs
        """
        feature_names = list(self.df.columns)
        pairs = (
            np.array(
                [
                    (feature_names[i], feature_names[j]) for i in range(len(feature_names)) \
                        for j in range(i + 1, len(feature_names))
                ]
            )
        )
        logger.info(f"{len(pairs)} feature pairs")
        return pairs

    # -------- Detection (Dask-parallel) --------
    def workflow(self, client):
        """
        Parallelized detection of outliers. Main driver function to run test. 

        Params:
        ------
            client: Dask distributed client
        Returns:
        ------
            df_density_scores: pd.DataFrame KDE density values for each cell across all feature pairs
        """
        rounds = self.n_rounds
        n_subsample = self.n_subsample 
        pairs = self.permute_pairs()
        logger.info("Starting KDE Estimation (Dask)")
        # Pre-extract and normalize features once to reduce serialization overhead
        x_cache = (
            {
                col: (self.df[col] / self.normalize_factor).to_numpy(dtype=np.float32, copy=False) \
                for col in self.df.columns
            }
        )
        delayed_tasks = []
        for metric_x, metric_y in pairs:
            x_vals = x_cache[metric_x]
            y_vals = x_cache[metric_y]
            task = (
                delayed(compute_median_density)(
                    metric_x, metric_y, x_vals, y_vals, rounds, n_subsample
                )
            )
            delayed_tasks.append(task)
        results = client.compute(delayed_tasks, sync=True)
        df_density_scores = pd.DataFrame(index=self.df.index)
        for col_name, med in results:
            df_density_scores[col_name] = med
        
        outlier_mask = self.identify_outliers(df_density_scores)
        self.plots(df_density_scores, outlier_mask)
        return 

    def identify_outliers(self, df_density_scores):
        # if the probability something is an outlier is less than the density threshold
        outlier_mask = (df_density_scores < self.density_thresh).any(axis=1)
        outlier_df = self.df[outlier_mask]
        outlier_df.to_csv(self.outlier_path)
        logger.info(f"Saved {len(outlier_df)} outliers to {self.outlier_path}")
        return outlier_mask

    # ---------- Plotting helpers (vector-first) ----------
    def _norm_df(self) -> pd.DataFrame:
        df_norm = self.df.copy()
        # selecting numeric columns
        num_cols = df_norm.select_dtypes(include=[np.number]).columns
        df_norm[num_cols] = df_norm[num_cols] / self.normalize_factor
        return df_norm

    def _save_vectorized(self, fig, plot_name: str):
        output_path = Path(self.figure_dir) if self.figure_dir else None
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                output_path / f"{plot_name}.svg",
                format="svg",
                bbox_inches="tight",
                facecolor="white",
            )
            logger.info(f"Saved plot: {output_path / f'{plot_name}.svg'}")
            return
        else:
            plt.show()

    def _clean_label(self, name: str) -> str:
        return name.replace("_", " ").replace("-", " ").title()

    # ---------- Plots ----------
    def outlier_summary(self, p_outlier, plot_name="outlier_summary"):
        colors = cc.glasbey_dark
        counts = (p_outlier < self.density_thresh).sum()
        labels = [c.replace("_vs_", " vs ").replace("-", " ").replace("_", " ") for c in p_outlier.columns]
        order = np.argsort(counts.values)[::-1] # reverse order
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.barh(np.array(labels)[order], counts.values[order], color=colors[0])
        ax.invert_yaxis()
        ax.set_xlabel("Number of Outliers")
        ax.set_title("Outliers per Feature Pair")
        fig.tight_layout()
        self._save_vectorized(fig, plot_name)


    def comprehensive_pairplot(self, outlier_mask, max_points=10000):
        df_plot = self.df_plot 
        df_plot["outlier_status"] = (
            pd.Categorical(
                np.where(outlier_mask, "Outlier", "Normal"),
                categories=["Normal", "Outlier"], 
                ordered=True
                )
        )
        # if len(df_plot) > max_points:
        #     logger.info(f"Sampling {max_points} points from {len(df_plot)}")
        #     df_plot = df_plot.sample(n=max_points, random_state=0)
        numeric_cols = df_plot.select_dtypes(include=[np.number]).columns.tolist()
        cols = numeric_cols + ["outlier_status"]
        g = sns.pairplot(
            df_plot[cols],
            hue="outlier_status",
            hue_order=["Normal", "Outlier"],  
            palette=[self.normal_color, self.outlier_color],
            plot_kws={"alpha": 0.6, "s": 10},
            diag_kind="kde",
        )
        g.fig.suptitle("Comprehensive Feature Pairplot", y=1.02, fontsize=16)
        self._save_vectorized(g, "pairplot")

        # if save_plots:
        #     output_path = Path(self.figure_dir) if self.figure_dir else None
        #     if output_path:
        #         output_path.mkdir(parents=True, exist_ok=True)
        #         logger.info("Saving comprehensive pairplot...")
        #         g.savefig(
        #             output_path / "comprehensive_pairplot.svg",
        #             format="svg",
        #             bbox_inches="tight",
        #             facecolor="white",
        #         )
        #         plt.close(g)
        #         logger.info(f"Saved plot: {output_path / 'comprehensive_pairplot.svg'}")
        #     else:
        #         plt.show()
        # else:
        #     plt.show()

    def plots(self, df_density_scores, outlier_mask):
        logger.info("Generating diagnostic plots...")
        self.df_plot = self._norm_df()  # Call the method to get the DataFrame
        self.outlier_summary(df_density_scores)
        self.comprehensive_pairplot(outlier_mask)


def _setup_dask(params):
    """Setup Dask SGE cluster for distributed processing."""
    try:
        logger.info("Setting up distributed Dask cluster...")
        cluster, client = setup_dask_sge_cluster(
            n_workers=params.get("num_workers", 1),
            cores=params.get("cpu_cores", 4),
            processes=params.get("cpu_processes", 1),
            memory=params.get("cpu_memory", "60G"),
            project=params.get("project", "beliveaulab"),
            queue=params.get("queue", "beliveau-long.q"),
            runtime=params.get("runtime", "7200"),
            resource_spec=params.get("cpu_resource_spec", "mfree=60G"),
            log_directory=params.get("log_dir", None),
            conda_env=params.get("conda_env", "otls-pipeline"),
            dashboard_port=params.get("dashboard_port", None),
        )
        logger.info(f"Dask dashboard link: {client.dashboard_link}")
        return cluster, client
    except Exception as e:
        logger.error(f"Failed to setup distributed cluster: {e}")
        return None, None


def _shutdown_dask(cluster, client):
    if cluster and client:
        try:
            shutdown_dask(cluster, client)
            logger.info("Dask cluster shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down cluster: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="KDE Outlier Detection (Dask-parallel)")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="CSV file with single cell data")
    parser.add_argument("--figure_dir", type=str, required=True,
                        help="Directory to store figures. Will be created if not exists")
    parser.add_argument("--outlier_path", type=str, required=True,
                        help="Where to store csv file of outliers")
    parser.add_argument("--density_thresh", type=float, default=1e-15,
                        help="Density threshold at which to consider an outlier")
    parser.add_argument("--n_subsample", type=int, default=10000,
                        help="Size of subsample in Monte Carlo subsampling")
    parser.add_argument("--n_rounds", type=int, default=5,
                        help="Num of rounds in KDE")
    add_dask_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    params = {
        "figure_dir": args.figure_dir,
        "density_thresh": args.density_thresh,
        "csv_path": args.csv_path,
        "outlier_path": args.outlier_path,
        "n_subsample": args.n_subsample,
        "n_rounds": args.n_rounds,
    }

    logger.info("--------------------------------")
    logger.info("Arguments read, starting analysis...")
    logger.info("--------------------------------")
    start_time = time.time()

    detector = KDEOutlierDetection(params)
    cluster = client = None
    try:
        dask_params = dask_config_from_args(args)
        cluster, client = _setup_dask(dask_params)
        logger.info(f"Dask dashboard: {client.dashboard_link}")
        detector.workflow(client)

    finally:
        _shutdown_dask(cluster, client)

    end_time = time.time()
    logger.info(f"Analysis completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
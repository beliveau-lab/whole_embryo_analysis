import numpy as np 
import polars as pl
from tifffile import imread
import plotly.graph_objects as go
import math
import skimage.measure as measure
from scipy.interpolate import splprep, splev
import plotly.graph_objects as go
from scipy.linalg import norm
import seaborn as sns
import matplotlib.pyplot as plt
from whole_embryo_analysis.somite_analysis.bio_frame import get_somite_centroids
import plotly.graph_objects as go
import colorcet as cc
from matplotlib.colors import ListedColormap

smoothing_colors = ListedColormap(cc.glasbey_dark[:12])

class SomiteSpline:
    def __init__(self, side, embryo_id, lut, mask, n_trim=0):
        self.lut = get_somite_centroids(mask, lut, scale=False)
        self.side = side
        self.embryo_id = embryo_id
        self.lut = (
            self.lut
            .filter(pl.col("Side") == side)
            .filter(pl.col("Embryo_ID") == embryo_id)
            .sort("Somite_Stage")
            .select(["Somite_ID", "Somite_Stage", "Centroid_Z", "Centroid_Y", "Centroid_X"])
        )
        if n_trim > 0:
            self.lut = (self.lut
                .filter(pl.col("Somite_Stage").is_between(min(self.lut["Somite_Stage"]) + n_trim, max(self.lut["Somite_Stage"]) - n_trim))
                .sort("Somite_Stage")
                .select(["Somite_ID", "Somite_Stage", "Centroid_Z", "Centroid_Y", "Centroid_X"])
            )
        self.inital_lengths = self.init_arc_lengths()
    
    def init_arc_lengths(self):
        centroids = (
            self.lut
            .select(["Centroid_Z", "Centroid_Y", "Centroid_X"])
            .to_numpy()
        )
        diffs = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
        return np.concatenate([[0], np.cumsum(diffs)])
    
    def fit_spline(self, lengths, smoothing=2.0, k=5):
        centroids = self.lut.select(["Centroid_Z", "Centroid_Y", "Centroid_X"]).to_numpy()
        tck, u = splprep(centroids.T, u=lengths, s=smoothing, k=k)
        return tck, u

    def evaluate_spline(self, u_values, tck):
        r = np.column_stack(splev(u_values, tck, der=0))
        dr = np.column_stack(splev(u_values, tck, der=1))
        ddr = np.column_stack(splev(u_values, tck, der=2))
        dddr = np.column_stack(splev(u_values, tck, der=3))
        return r, dr, ddr, dddr

    def calculate_kappa(self, dr, ddr):
        cross = np.cross(dr, ddr)
        kappas = np.linalg.norm(cross, axis=1) / (np.linalg.norm(dr, axis=1) ** 3)
        return kappas

    def calculate_tau(self, dr, ddr, dddr):
        cross = np.cross(dr, ddr)
        numerator = np.sum(cross * dddr, axis=1)
        denominator = np.linalg.norm(cross, axis=1) ** 2
        taus = np.where(denominator < 1e-10, np.nan, numerator / denominator)
        return taus

    def correct_arc_lengths(self, smoothing=2.5, tolerance=1e-10):
        centroids = (
            self.lut
            .select(["Centroid_Z", "Centroid_Y", "Centroid_X"])
            .to_numpy()
        )
        lengths = self.inital_lengths.copy()
        i = 0
        while True:
            tck, u_somites = self.fit_spline(lengths, smoothing=smoothing, k=5)
            t_fine = np.linspace(0, max(lengths), 10000)
            # evaluate spline derivative at many fine points
            derivatives = np.column_stack(splev(t_fine, tck, der=1))
            # magnitude of derivative at each fine point (rate of change of arc length)
            speeds = np.linalg.norm(derivatives, axis=1)
            # integrate to get true curve length
            L_new = np.trapezoid(speeds, t_fine)
            # compare to current total arc length
            l_n = lengths[-1]
            # if the difference is less than the tolerance, break
            if np.abs(1 - l_n / L_new) < tolerance:
                # print(f"Converged after {i} iterations")
                break
            # else rescale all arc lengths proportionally
            lengths = lengths * (L_new / l_n)
            i += 1
            if i > 100:
                print("Hasn't converged after 100 iterations")
                break
        return lengths, tck, u_somites
        
    def determine_smoothing(self):
        smoothing_values = np.logspace(-2, 2, 10)
        all_kappas = []
        for smoothing in smoothing_values:
            lengths, tck, u_somites = self.correct_arc_lengths(smoothing=smoothing)
            r, dr, ddr, dddr = self.evaluate_spline(u_somites, tck)
            kappas = self.calculate_kappa(dr, ddr)
            all_kappas.append(kappas)
        return smoothing_values, np.array(all_kappas)

    def plot_smoothing_sweep(self, smoothing_values, all_kappas):
        fig, ax = plt.subplots(figsize=(10, 5))
        stages = self.lut.select("Somite_Stage").to_numpy().flatten()
        for i, s in enumerate(smoothing_values):
            ax.plot(stages, all_kappas[i], marker='.', label=f's={s:.3f}', alpha=0.7, color=smoothing_colors(i))
        ax.set_xlabel('Somite Stage')
        ax.set_ylabel('Curvature (κ)')
        ax.set_title(f'Curvature Sensitivity to Smoothing\n Sample: {self.embryo_id}, Side: {self.side}')
        ax.set_xticks(stages)
        ax.set_xticklabels(stages)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax.legend(title='Smoothing Value', fontsize=8)
        plt.show()
        return

    def compute_stage_values(self, smoothing=1.0, visualize_spline=False):
        lengths, tck, u_somites = self.correct_arc_lengths(smoothing=smoothing)
        r, dr, ddr, dddr = self.evaluate_spline(u_somites, tck)
        kappas = self.calculate_kappa(dr, ddr)
        taus = self.calculate_tau(dr, ddr, dddr)
        stages = self.lut.select("Somite_Stage").to_numpy().flatten()
        # global normalizer: mean inter-somite spacing for this embryo
        mean_spacing = lengths[-1] / (len(stages) - 1)
        if visualize_spline:
            self.plot_spline_3d(u_somites, tck)
        return pl.DataFrame({
            "Somite_Stage": stages,
            "Kappa": kappas,
            "Tau": taus,
            "Kappa_Norm": kappas * mean_spacing,
            "Tau_Norm": taus * mean_spacing,
        })


    def plot_spline_3d(self, u_somites, tck):
        fig = go.Figure()
        r, _, _, _ = self.evaluate_spline(u_somites, tck)
        centroids = self.lut.select(["Centroid_Z", "Centroid_Y", "Centroid_X"]).to_numpy()
        fig.add_trace(go.Scatter3d(
            x=r[:, 0],
            y=r[:, 1],
            z=r[:, 2],
            mode='lines',
            line=dict(color='black', width=4),
            name="Spline",
        ))
        fig.add_trace(go.Scatter3d(
            x=centroids[:, 0],
            y=centroids[:, 1],
            z=centroids[:, 2],
            mode='markers',
            marker=dict(color='red', size=4),
            name="Centroids",
        ))
        fig.update_layout(
            scene=dict(
                aspectmode="data",
            ),
            title=f"Spline for Embryo {self.embryo_id}, Side: {self.side}",
        )
        fig.show()


def run_embryo(embryo_ID, lut, mask, smoothing_parameter=0.5, visualize_spline=False, n_trim=0):
    left_spline = (
        SomiteSpline(side="Left", lut=lut, mask=mask, embryo_id=embryo_ID, n_trim=n_trim)
        )
    right_spline = (
        SomiteSpline(side="Right", lut=lut, mask=mask, embryo_id=embryo_ID, n_trim=n_trim)
        )
    Ldf = left_spline.compute_stage_values(smoothing=smoothing_parameter, visualize_spline=visualize_spline)
    Rdf = right_spline.compute_stage_values(smoothing=smoothing_parameter, visualize_spline=visualize_spline)
    merged = Ldf.join(Rdf, on="Somite_Stage", suffix="_R")
    merged = (
        merged
        .with_columns([
            pl.lit(embryo_ID).alias("Embryo_ID"),
            (pl.col("Kappa") - pl.col("Kappa_R")).alias("Delta_Kappa"),
            (pl.col("Tau") - pl.col("Tau_R")).alias("Delta_Tau"),
            (pl.col("Kappa_Norm") - pl.col("Kappa_Norm_R")).alias("Delta_Kappa_Norm"),
            (pl.col("Tau_Norm") - pl.col("Tau_Norm_R")).alias("Delta_Tau_Norm")
        ])
    )
    return merged

def plot_assymetry(df, metric):
    embryo_id = df.select("Embryo_ID").to_numpy().flatten()[0]
    if metric == "Kappa":
        string = "κ"
        title = "Curvature"
    else:
        string = "τ"
        title = "Torsion"
    fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    stages = df.select("Somite_Stage").to_numpy().flatten()
    # top: L and R kappa overlaid
    axes[0].plot(stages, df[f"{metric}"].to_numpy(), 'b.-', label='Left')
    axes[0].plot(stages, df[f"{metric}_R"].to_numpy(), 'r.-', label='Right')
    axes[0].set_ylabel(f'{title} ({string})')
    axes[0].set_title(f'{metric.capitalize()} ({string}) by Stage \n Embryo: {embryo_id}')
    axes[0].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axes[0].set_xticks(stages)
    axes[0].set_xticklabels(stages)
    axes[0].legend()
    # bottom: difference
    delta = df[f"Delta_{metric}"].to_numpy()
    axes[1].bar(stages, delta, color=np.where(delta > 0, 'steelblue', 'salmon'))
    axes[1].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axes[1].set_ylabel(f'Δ{string} (L − R)')
    axes[1].set_xlabel('Somite Stage')
    axes[1].set_xticks(stages)
    plt.tight_layout()
    plt.show()
    return
    


# def aggregate_across_embryos(embryo_data, smoothing=1.0):
#     """
#     Pool asymmetry across embryos with standardized curvature.
    
#     embryo_data: list of dicts, each with keys 'left_spline', 'right_spline', 'embryo_id'
#     """
#     all_asymmetries = []
#     for embryo in embryo_data:
#         left_df = compute_stage_values_standardized(embryo['left_spline'], smoothing)
#         right_df = compute_stage_values_standardized(embryo['right_spline'], smoothing)
        
#         merged = left_df.join(right_df, on="Somite_Stage", suffix="_R")
#         asymmetry = merged.with_columns([
#             (pl.col("kappa_std") - pl.col("kappa_std_R")).alias("delta_kappa_std"),
#             (pl.col("tau_std") - pl.col("tau_std_R")).alias("delta_tau_std"),
#             pl.lit(embryo['embryo_id']).alias("Embryo_ID"),
#         ])
#         all_asymmetries.append(asymmetry)
    
#     pooled = pl.concat(all_asymmetries)
#     return pooled

# def summarize_by_stage(pooled):
#     """Mean and SEM of asymmetry per stage across embryos."""
#     summary = pooled.group_by("Somite_Stage").agg([
#         pl.col("delta_kappa_std").mean().alias("mean_delta_kappa"),
#         pl.col("delta_kappa_std").std().alias("std_delta_kappa"),
#         pl.col("delta_kappa_std").count().alias("n"),
#         pl.col("delta_tau_std").mean().alias("mean_delta_tau"),
#         pl.col("delta_tau_std").std().alias("std_delta_tau"),
#     ]).sort("Somite_Stage").with_columns([
#         (pl.col("std_delta_kappa") / pl.col("n").sqrt()).alias("sem_delta_kappa"),
#         (pl.col("std_delta_tau") / pl.col("n").sqrt()).alias("sem_delta_tau"),
#     ])
#     return summary

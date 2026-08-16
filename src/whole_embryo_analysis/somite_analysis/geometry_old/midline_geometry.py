import numpy as np
import polars as pl
from scipy.interpolate import splprep, splev
from whole_embryo_analysis.somite_analysis.geometry.curvature import SomiteSpline
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

class MidlineSpline:
    def __init__(self, embryo_id, lut, mask, smoothing=1.7, n_trim=1):
        self.embryo_id = embryo_id
        self.left = SomiteSpline(side="Left", lut=lut, mask=mask, embryo_id=embryo_id, n_trim=n_trim)
        self.right = SomiteSpline(side="Right", lut=lut, mask=mask, embryo_id=embryo_id, n_trim=n_trim)

        # only keep stages present on both sides
        left_stages = set(self.left.lut["Somite_Stage"].to_list())
        right_stages = set(self.right.lut["Somite_Stage"].to_list())
        shared = sorted(left_stages & right_stages)
        self.left.lut = self.left.lut.filter(pl.col("Somite_Stage").is_in(shared))
        self.right.lut = self.right.lut.filter(pl.col("Somite_Stage").is_in(shared))

        self.stages = np.array(shared)
        self.smoothing = smoothing

        left_centroids = self.left.lut.select(
            ["Centroid_Z", "Centroid_Y", "Centroid_X"]
        ).to_numpy()
        right_centroids = self.right.lut.select(
            ["Centroid_Z", "Centroid_Y", "Centroid_X"]
        ).to_numpy()
        self.left_centroids = left_centroids
        self.right_centroids = right_centroids
        self.midline_centroids = (left_centroids + right_centroids) / 2

    def init_arc_lengths(self):
        diffs = np.linalg.norm(np.diff(self.midline_centroids, axis=0), axis=1)
        return np.concatenate([[0], np.cumsum(diffs)])

    def fit_midline(self, smoothing=None, tolerance=1e-7):
        """Fit spline to midline and correct arc lengths."""
        if smoothing is None:
            smoothing = self.smoothing
        lengths = self.init_arc_lengths()
        for i in range(100):
            tck, u_somites = splprep(self.midline_centroids.T, u=lengths, s=smoothing, k=5)
            t_fine = np.linspace(0, lengths[-1], 10000)
            derivatives = np.column_stack(splev(t_fine, tck, der=1))
            speeds = np.linalg.norm(derivatives, axis=1)
            L_new = np.trapezoid(speeds, t_fine)
            l_n = lengths[-1]
            if np.abs(1 - l_n / L_new) < tolerance:
                break
            lengths = lengths * (L_new / l_n)
        else:
            print("Hasn't converged after 100 iterations")
        return lengths, tck, u_somites

    def evaluate_midline(self, u_values, tck):
        r = np.column_stack(splev(u_values, tck, der=0))
        dr = np.column_stack(splev(u_values, tck, der=1))
        ddr = np.column_stack(splev(u_values, tck, der=2))
        dddr = np.column_stack(splev(u_values, tck, der=3))
        return r, dr, ddr, dddr

    def calculate_tangent(self, dr):
        return dr / np.linalg.norm(dr, axis=1, keepdims=True)

    def calculate_kappa(self, dr, ddr):
        cross = np.cross(dr, ddr)
        return np.linalg.norm(cross, axis=1) / (np.linalg.norm(dr, axis=1) ** 3)

    def calculate_normal(self, ddr, tangents):
        proj = np.sum(ddr * tangents, axis=1, keepdims=True) * tangents
        normals_raw = ddr - proj
        norms = np.linalg.norm(normals_raw, axis=1, keepdims=True)
        return np.where(norms > 1e-10, normals_raw / norms, 0)

    def determine_convex_side(self, midline_points, normals):
        """At each stage, determine which side is convex."""
        to_left = self.left_centroids - midline_points
        dot_left = np.sum(normals * to_left, axis=1)
        convex_side = np.where(dot_left > 0, "Right", "Left")
        return convex_side, dot_left

    def build_results(self, volume_df, kappas, convex_side, dot_left):
        """Combine with volume data and test prediction."""
        vol_curr = volume_df.filter(pl.col("Embryo_ID") == self.embryo_id)

        results = pl.DataFrame({
            "Embryo_ID": [self.embryo_id] * len(self.stages),
            "Somite_Stage": self.stages,
            "Midline_Kappa": kappas,
            "Convex_Side": convex_side.tolist(),
            "Normal_Dot_Left": dot_left,
        })

        results = results.join(vol_curr, on=["Embryo_ID", "Somite_Stage"], how="left")

        results = results.with_columns(
            pl.when(pl.col("Percent_Delta_Vol") > 0)
            .then(pl.lit("Left"))
            .otherwise(pl.lit("Right"))
            .alias("Bigger_Side")
        )

        results = results.with_columns(
            (pl.col("Convex_Side") == pl.col("Bigger_Side")).alias("Convex_Is_Bigger")
        )
        return results

    def workflow(self, volume_df, inspect_smoothing=False, visualize_spline=False):
        if inspect_smoothing:
            self.inspect_smoothing()
            return
        lengths, tck, u_somites = self.fit_midline()
        r, dr, ddr, dddr = self.evaluate_midline(u_somites, tck)
        tangents = self.calculate_tangent(dr)
        kappas = self.calculate_kappa(dr, ddr)
        normals = self.calculate_normal(ddr, tangents)
        convex_side, dot_left = self.determine_convex_side(r, normals)
        if visualize_spline:
            # self.plot_kappa_profile(kappas)
            self.plot_spline_3d(u_somites, tck)
        return self.build_results(volume_df, kappas, convex_side, dot_left)

    def determine_smoothing(self):
        smoothing_values = np.logspace(-2, 2, 10)
        all_kappas = []
        for smoothing in smoothing_values:
            lengths, tck, u_somites = self.fit_midline(smoothing=smoothing)
            r, dr, ddr, dddr = self.evaluate_midline(u_somites, tck)
            kappas = self.calculate_kappa(dr, ddr)
            all_kappas.append(kappas)
        return smoothing_values, np.array(all_kappas)

    def plot_smoothing_sweep(self, smoothing_values, all_kappas):
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, s in enumerate(smoothing_values):
            ax.plot(self.stages, all_kappas[i], marker='.', label=f's={s:.3f}', alpha=0.7)
        ax.set_xlabel('Somite Stage')
        ax.set_ylabel('Curvature (κ)')
        ax.set_title(f'Midline Curvature Sensitivity to Smoothing\nEmbryo: {self.embryo_id}')
        ax.set_xticks(self.stages)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax.legend(title='Smoothing Value', fontsize=8)
        plt.tight_layout()
        plt.show()
    
    def inspect_smoothing(self):
        smoothing_values, all_kappas = self.determine_smoothing()
        self.plot_smoothing_sweep(smoothing_values, all_kappas)
        return

    def plot_spline_3d(self, u_somites, tck):
        fig = go.Figure()
        r, _, _, _ = self.evaluate_midline(u_somites, tck)
        fig.add_trace(go.Scatter3d(
            x=r[:, 0],
            y=r[:, 1],
            z=r[:, 2],
            mode='lines',
            line=dict(color='black', width=4),
            name="Midline",
        ))
        # add left centroids
        fig.add_trace(go.Scatter3d(
            x=self.left_centroids[:, 0],
            y=self.left_centroids[:, 1],
            z=self.left_centroids[:, 2],
            mode='markers',
            marker=dict(color='red', size=4),
            name="Left Centroids",
        ))
        # add right centroids
        fig.add_trace(go.Scatter3d(
            x=self.right_centroids[:, 0],
            y=self.right_centroids[:, 1],
            z=self.right_centroids[:, 2],
            mode='markers',
            marker=dict(color='blue', size=4),
            name="Right Centroids",
        ))

        fig.update_layout(
            scene=dict(
                aspectmode="data",
            ),
            title=f"Midline Spline for Embryo {self.embryo_id}",
        )
        fig.show()
        return

    def plot_kappa_profile(self, kappas):
        fig, axes = plt.subplots(1, 1, figsize=(8, 4), sharex=True)
        sns.lineplot(x=self.stages, y=kappas, ax=axes, color="black")
        axes.set_title(f"Midline Curvature Profile for Embryo {self.embryo_id}")
        axes.set_xlabel("Somite Stage")
        axes.set_ylabel("Midline Curvature")
        axes.set_xticks(self.stages)
        axes.set_xticklabels(self.stages)
        plt.show()
        return

# helper functions
def run_embryo(embryo_id, lut, mask, volume_df, smoothing=0.5, inspect_smoothing=False, visualize_spline=False, n_trim=3):
    if embryo_id == 324:
        # print(f"Embryo {embryo_id}: trimming 2 stages")
        # visualize_spline = True
        # n_trim = 2
    mid = MidlineSpline(embryo_id, lut, mask, smoothing=smoothing, n_trim=n_trim)
    if len(mid.stages) < 6:
        print(f"Embryo {embryo_id}: insufficient shared stages ({len(mid.stages)})")
        return None
    return mid.workflow(volume_df, inspect_smoothing=inspect_smoothing, visualize_spline=visualize_spline)

def plot_midline_prediction(results, n_trim=2):
    """Visualize whether convex side predicts bigger side."""
    stages = results["Somite_Stage"].to_numpy()
    min_s, max_s = stages.min(), stages.max()
    trimmed = results.filter(
        pl.col("Somite_Stage").is_between(min_s + n_trim, max_s - n_trim)
    )
    embryo_id = trimmed["Embryo_ID"][0]
    stages = trimmed["Somite_Stage"].to_numpy()
    kappa = trimmed["Midline_Kappa"].to_numpy()
    vol_diff = trimmed["Percent_Delta_Vol"].to_numpy()
    correct = trimmed["Convex_Is_Bigger"].to_numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # midline curvature
    ax1.bar(stages, kappa, color='grey')
    ax1.set_ylabel('Midline κ')
    ax1.set_title(f'Embryo {embryo_id}: Midline curvature predicting volume asymmetry')

    # volume asymmetry
    ax2.bar(stages, vol_diff, color=np.where(vol_diff > 0, 'steelblue', 'salmon'))
    ax2.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax2.set_ylabel('Volume % Diff (L−R)')

    # prediction accuracy
    colors = np.where(correct, 'green', 'red')
    ax3.bar(stages, np.ones(len(stages)), color=colors)
    ax3.set_yticks([])
    ax3.set_ylabel('Convex = Bigger?')
    ax3.set_xlabel('Somite Stage')
    ax3.set_xticks(stages)

    # summary
    acc = correct.sum() / len(correct)
    ax3.set_title(f'Accuracy: {correct.sum()}/{len(correct)} = {acc:.1%}')

    plt.tight_layout()
    plt.show()



# def plot_pooled_prediction(pooled, n_trim=1):
#     """Across all embryos: fraction where convex side is bigger, per stage."""
#     min_s = pooled["Somite_Stage"].min()
#     max_s = pooled["Somite_Stage"].max()
#     print(f"Pooled prediction: {min_s} to {max_s}")
#     trimmed = pooled.filter(
#         pl.col("Somite_Stage").is_between(min_s + n_trim, max_s - n_trim)
#     )

#     summary = (
#         trimmed
#         .group_by("Somite_Stage")
#         .agg([
#             pl.col("Convex_Is_Bigger").sum().alias("n_correct"),
#             pl.col("Convex_Is_Bigger").count().alias("n_total"),
#         ])
#         .sort("Somite_Stage")
#         .with_columns(
#             (pl.col("n_correct") / pl.col("n_total")).alias("accuracy"),
#         )
#     )

#     stages = summary["Somite_Stage"].to_numpy()
#     acc = summary["accuracy"].to_numpy()
#     n_correct = summary["n_correct"].to_numpy()
#     n_total = summary["n_total"].to_numpy()

#     fig, ax = plt.subplots(figsize=(12, 5))
#     colors = np.where(acc > 0.5, 'green', 'red')
#     ax.bar(stages, acc, color=colors, alpha=0.7)
#     ax.axhline(0.5, color='grey', linestyle='--', linewidth=1, label='Chance')

#     for i, s in enumerate(stages):
#         ax.text(s, acc[i] + 0.02, f'{int(n_correct[i])}/{int(n_total[i])}',
#                 ha='center', fontsize=8)

#     ax.set_xlabel('Somite Stage')
#     ax.set_ylabel('Fraction: convex side is bigger')
#     ax.set_title(f'Does embryonic bending predict which side is bigger? (n={trimmed["Embryo_ID"].n_unique()} embryos)')
#     ax.set_xticks(stages)
#     ax.set_ylim(0, 1.1)
#     ax.legend()
#     plt.tight_layout()
#     plt.show()

#     # overall binomial test
#     from scipy.stats import binomtest
#     total_correct = int(trimmed["Convex_Is_Bigger"].sum())
#     total_n = len(trimmed)
#     result = binomtest(total_correct, total_n, 0.5, alternative='greater')
#     print(f"Overall: {total_correct}/{total_n} = {total_correct/total_n:.1%}, p = {result.pvalue:.4f}")

def plot_pooled_prediction(pooled, n_trim=2):
    min_s = pooled["Somite_Stage"].min()
    max_s = pooled["Somite_Stage"].max()
    trimmed = pooled.filter(
        pl.col("Somite_Stage").is_between(min_s + n_trim, max_s - n_trim)
    )

    summary = (
        trimmed
        .group_by("Somite_Stage")
        .agg([
            pl.col("Convex_Is_Bigger").sum().alias("n_convex"),
            pl.col("Convex_Is_Bigger").count().alias("n_total"),
        ])
        .sort("Somite_Stage")
        .with_columns(
            (pl.col("n_convex") / pl.col("n_total")).alias("frac_convex"),
        )
    )

    stages = summary["Somite_Stage"].to_numpy()
    frac = summary["frac_convex"].to_numpy()
    n_convex = summary["n_convex"].to_numpy().astype(int)
    n_total = summary["n_total"].to_numpy().astype(int)

    # deviation from chance: positive = convex bigger, negative = concave bigger
    deviation = frac - 0.5

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = np.where(deviation > 0, 'steelblue', 'salmon')
    ax.bar(stages, deviation, color=colors)
    ax.axhline(0, color='black', linewidth=1)

    for i, s in enumerate(stages):
        label = f'{n_convex[i]}/{n_total[i]}'
        y_pos = deviation[i] + (0.02 if deviation[i] >= 0 else -0.04)
        ax.text(s, y_pos, label, ha='center', fontsize=8)

    ax.set_xlabel('Somite Stage')
    ax.set_ylabel('← Concave bigger | Convex bigger →')
    ax.set_title(f'Does embryonic bending predict which side is bigger? (n={trimmed["Embryo_ID"].n_unique()} embryos)')
    ax.set_xticks(stages)
    ax.set_ylim(-0.6, 0.6)
    plt.tight_layout()
    plt.show()

    from scipy.stats import binomtest
    total_correct = int(trimmed["Convex_Is_Bigger"].sum())
    total_n = len(trimmed)
    result = binomtest(total_correct, total_n, 0.5, alternative='greater')
    print(f"Overall: {total_correct}/{total_n} = {total_correct/total_n:.1%}, p = {result.pvalue:.4f}")
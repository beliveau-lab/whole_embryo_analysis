
import os
from pathlib import Path
import logging

import pyvista as pv
import matplotlib.pyplot as plt
from numpy import arange 
from sklearn.decomposition import PCA

import polars as pl
import numpy as np
from colorcet import glasbey_dark
from aicsshparam.shtools import (get_reconstruction_from_coeffs, 
                                save_polydata, 
                                convert_coeffs_dict_to_matrix)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def hex_to_rgb(hex):
    hex = hex.lstrip('#')
    return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))

NUCLEAR_COLOR_RGB = hex_to_rgb(glasbey_dark[43])

def save_vectorized(dir, fig, plot_name):
    output_path = Path(dir) if dir else None
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

def get_sigma_steps():
    max_sd = 2.0
    num_points = int(2 * max_sd) + 1
    return np.linspace(-max_sd, max_sd, num_points).tolist()


def mesh_viz_small(shapespace):
    sigma_steps_desired = np.arange(-2, 3, 1)
    fixed_pos = 65
    fixed_position = [fixed_pos, fixed_pos, fixed_pos]
    fixed_focal_point = [0, 0, 0]
    num_pcs = shapespace.num_pcs
    plotter = pv.Plotter(shape=(num_pcs, len(sigma_steps_desired)), 
                    window_size=(2000, 1000), 
                    off_screen=True,
                    lighting='three_lights')
    for pc in range(1, num_pcs+1):
        pc_key = f"PC{pc}"
        col_index = 0  # Track column index separately
        for s_index, sigma in enumerate(shapespace.sigma_steps):
            if sigma.is_integer():
                mesh = pv.read(f"{shapespace.save_path}/meshes/{pc_key}_vtks/{pc_key}_{sigma}.vtk")
            # mesh = shapespace.meshes[pc_key][s_index]
                plotter.subplot(pc-1, col_index)
                plotter.add_mesh(mesh, color='steelblue')
                plotter.camera.position = fixed_position
                plotter.camera.focal_point = fixed_focal_point 
                col_index += 1  # Only increment when we actually add a mesh
            else:
                continue
    if shapespace.store_results:
        plotter.save_graphic(f"{shapespace.save_path}/reconstruction_viz_small.svg")
    else: 
        plotter.show()
    return plotter

def mesh_visualization(shapespace):
    fixed_pos = 65
    fixed_position = [fixed_pos, fixed_pos, fixed_pos]
    fixed_focal_point = [0, 0, 0]
    num_pcs = shapespace.num_pcs
    plotter = pv.Plotter(shape=(num_pcs, len(shapespace.sigma_steps)), 
                    window_size=(2000, 1000), 
                    off_screen=True,
                    lighting='three_lights')
    for pc in range(1, num_pcs+1):
        pc_key = f"PC{pc}"
        for s_index, _ in enumerate(shapespace.sigma_steps):
            # mesh = pv.read(f"{mesh_path}/{pc_key}_vtks/{pc_key}_{shapespace.sigma_steps[s_index]}.vtk")
            mesh = shapespace.meshes[pc_key][s_index]
            plotter.subplot(pc-1, s_index)
            plotter.add_mesh(mesh, color='steelblue')
            plotter.camera.position = fixed_position
            plotter.camera.focal_point = fixed_focal_point 
    if shapespace.store_results:
        plotter.save_graphic(f"{shapespace.save_path}/reconstruction_viz.svg")
    else: 
        plotter.show()

    return plotter


def reconstruct_meshes(shapespace):
    """
    Reconstruct meshes for each shaoemode   
    
    Params:
    ------
        save_dir: str of directory path to save meshes
        str_tag: str identifier tag
    """
    reconstruction_df = shapespace.shapespace_to_shcoeff()
    meshes = {}
    meshes_path = f"{shapespace.save_path}/meshes"
    # first directroy will be /meshes
    os.makedirs(meshes_path, exist_ok=True)
    for _, df_sm in reconstruction_df.groupby("shape_mode"):
        cur_pc = df_sm['shape_mode'].values[0]
        shapemode_path = f"{meshes_path}/{cur_pc}_vtks"
        meshes[cur_pc] = []
        # inside meshes will be/PC#_vtks/
        os.makedirs(shapemode_path, exist_ok=True)
        for _, row in df_sm.iterrows():
            shcoeffs = row.filter(like="shcoeffs_")
            row_dict = {coeff_name: shcoeffs[coeff_name] for coeff_name in shapespace.shcoeffs_df.columns}
            coeffs = convert_coeffs_dict_to_matrix(row_dict, lmax=shapespace.lmax)
            mesh, _ = get_reconstruction_from_coeffs(coeffs)
            meshes[cur_pc].append(mesh)
            # inside meshes/PC_vtks/ will be the vtk files for each principal component ranging from -2σ to +2σ
            if shapespace.store_results:
                save_polydata(mesh, f"{shapemode_path}/{row['shape_mode']}_{shapespace.sigma_steps[int(row['mpId'])-1]}.vtk")
    print(f"All meshes saved to: {shapespace.save_path}")
    shapespace.meshes = meshes
    return

def get_simga_labels(sigma_steps):
    labels = []
    for step in sigma_steps:
        if step == 0:
            string = f'$\\mu$'
        else:
            # Format as integer if it's a whole number, otherwise keep decimal
            if step == int(step):
                string = f'${int(step)}\\sigma$'
            else:
                string = f'${step}\\sigma$'
        labels.append(string)
    return labels

def pc_histograms(shapespace, font_size=10):
    num_pcs = shapespace.num_pcs 
    fig, ax = plt.subplots(num_pcs, 1, figsize=(4, 6))
    for pc in range(1, num_pcs+1):
        pc_idx = pc - 1
        data = shapespace.df_shapespace[f"PC{pc}_mpId"].value_counts(normalize=True) * 100
        # Use matplotlib bar instead of seaborn
        ax[pc_idx].bar(data.index, data.values, color='gray')
        ax[pc_idx].grid(False)
        # Remove x-axis labels for all but the bottom subplot
        if pc_idx < num_pcs - 1:
            ax[pc_idx].set_xlabel(f'PC{pc}')
            # ax[pc_idx].set_xticklabels()
        else:
            ax[pc_idx].set_xticks(arange(len(shapespace.sigma_steps)))
            ax[pc_idx].set_xticklabels(get_simga_labels(shapespace.sigma_steps), fontsize=font_size)
            ax[pc_idx].set_xlabel("Standard Deviation Bins", fontsize=font_size)
    # Add shared y-axis label
    fig.text(0.04, 0.5, "Frequency of cells (%) per Principal Component", va='center', rotation='vertical', fontsize=font_size)
    plt.tight_layout()
    plt.subplots_adjust(left=0.15)  # Make room for the y-axis label
    if shapespace.store_results:
        save_vectorized(
            dir=shapespace.save_path,
            fig=fig,
            plot_name=f"shp_histogram"
        )
    else:
        plt.show()

def variance_explained(shapespace, font_size=10):
    num_pcs = shapespace.num_pcs
    fig, ax = plt.subplots(num_pcs, 1, figsize=(3, 5))
    variance_ratios = 100 * shapespace.pca.explained_variance_ratio_[:shapespace.num_pcs]
    # Create a horizontal bar in each subplot
    for pc in range(num_pcs):
        ax[pc].barh(0, variance_ratios[pc], color='black')
        ax[pc].set_yticks([])
        ax[pc].set_ylabel("")
        # Set x-axis limits to be consistent across all subplots
        ax[pc].set_xlim(0, variance_ratios.max() * 1.1)
        ax[pc].grid(False)
        # Remove x-axis labels for all but the bottom subplot
        if pc < num_pcs - 1:
            ax[pc].set_xlabel('')
            ax[pc].set_xticklabels([])
        else:
            ax[pc].set_xlabel("Variance Explained (%)", fontsize=font_size)
    if shapespace.store_results:
        save_vectorized(
            dir=shapespace.save_path,
            fig=fig,
            plot_name=f"variance_explained"
        )
    else:
        plt.show()

def plot_scree(shapespace, font_size=18, n_components=25):
    sh_matrix = shapespace.shcoeffs_df.values
    pca_full = PCA(n_components=n_components, svd_solver='full')
    pca_full.fit(sh_matrix)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    var_explained_chose = shapespace.pca.explained_variance_ratio_.sum()
    fig, ax = plt.subplots(figsize=(6, 8)) 
    ax.plot(cumulative_variance,    
            range(1, len(cumulative_variance) + 1), 
            'o-', 
            linewidth=2, 
            markersize=8,
            color='steelblue')
    
    ax.axhline(y=shapespace.num_pcs, color='red', linestyle='--', 
               label=f'Chosen: {shapespace.num_pcs} PCs')
    ax.grid(True)
    ax.legend(loc='best', fontsize=font_size)
    ax.set_xlabel('Var Explained (%)', fontsize=font_size)
    ax.set_title(f'Total Variance Explained: {var_explained_chose:.2f}%', fontsize=font_size)
    ax.set_xlim(cumulative_variance[0] - 0.05, 1.0)
    ax.set_ylim(0, len(cumulative_variance) + 1)
    ax.invert_yaxis()
    ax.set_ylabel("# of PCs", rotation=45, fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    if shapespace.store_results:
        save_vectorized(dir=shapespace.save_path, fig=fig, plot_name='pca_scree_plot')
    else:
        plt.show()
    
    return fig

# def create_mesh_subplots(mesh_path, pc, sigma_steps):
#     n = len(sigma_steps)
#     fig = make_subplots(rows=1, cols=n, specs=[[{'type': 'mesh3d'}] * n])
    
#     for i, sigma in enumerate(sigma_steps):
#         mesh = pv.read(f"{mesh_path}/PC{pc}_vtks/PC{pc}_{sigma}.vtk")
#         faces = mesh.faces.reshape(-1, 4)[:, 1:4]
#         fig.add_trace(go.Mesh3d(
#             x=mesh.points[:, 0], y=mesh.points[:, 1], z=mesh.points[:, 2],
#             i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
#             color="#1f77b4", lighting=dict(ambient=0.5, diffuse=0.5),
#             showscale=False, hoverinfo='skip'
#         ), row=1, col=i + 1)
    
#     scene = dict(
#         camera=dict(eye=dict(x=1, y=1, z=1), up=dict(x=0, y=0, z=1)),
#         xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
#         aspectmode='cube', bgcolor='white'
#     )
    
#     fig.update_layout(
#         **{('scene' if i == 1 else f'scene{i}'): scene for i in range(1, n + 1)},
#         height=120, width=80 * n, showlegend=False, margin=dict(l=5, r=5, t=5, b=5)
#     )
#     return fig


#  def load_data(self, full_path, outlier_path):
#     print("Loading data with Polars lazy loading...")
#     # Start with lazy scan - no data loaded yet
#     query = pl.scan_csv(full_path)
#     if outlier_path is not None:
#         outlier_ids = (
#             pl.read_csv(outlier_path)
#             .select(self.id_name)
#             .to_series()
#             .to_list()
#         )
#         query = (
#             query.filter(~pl.col(self.id_name).is_in(outlier_ids))
#             )
#     # Filter by column if given
#     if self.class_column and self.class_value:
#         query = query.filter(pl.col(self.class_column) == self.class_value)
#     # Only select columns we actually need
#     schema = query.collect_schema()
#     shcoeff_cols = [c for c in schema.names() if 'shcoeffs_' in c]
#     needed_cols = [self.id_name] + shcoeff_cols
#     # Execute query and collect only what's needed
#     df = query.select(needed_cols).collect()
#     print(f"Filtered to {len(df)} samples")
#     # Store for later use
#     self.label_index = df[self.id_name].to_pandas()
#     # Convert to pandas only for PCA computation (sklearn requires numpy/pandas format)
#     self.shcoeffs_df = (
#         df.select(shcoeff_cols)
#         .to_pandas()
#         .set_index(self.label_index)
#     )
#     return df
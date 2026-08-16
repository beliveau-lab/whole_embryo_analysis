import logging
from pathlib import Path

# Core imports - used across many functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec

from whole_embryo_analysis.utils.scale_resolution import S3_SCALE, from_s0
from whole_embryo_analysis.utils.viz import hex_to_rgb, save_vectorized, napari_camera_to_plotly

logger = logging.getLogger(__name__)

CLUSTER_COLORS = {
    1: "#FF00FF",  # magenta
    2: "#00FFFF",  # cyan
    3: "#4daf4a",  # green
    4: "#e41a1c",  # red
    5: "#377eb8",  # blue
    6: "#ffff33",  # yellow
    7: "#a65628",  # brown
    8: "#f781bf",  # pink
}


def get_cluster_colors():
    return CLUSTER_COLORS
    
def plot_all_somite_masks(somite_mask, all_somites, highlight_tup=None):
    """Plot somite mask MIP with optional highlighting of a specific somite.
    
    Args:
        somite_mask: 3D labeled mask array
        all_somites: DataFrame with Somite_ID and somite_stage columns
        highlight_id: Somite ID to highlight in a different color (optional)
    """
    import colorcet as cc
    
    somite_id_to_stage = dict(zip(all_somites['Somite_ID'], all_somites['somite_stage']))
    
    # Create MIP
    mip = np.max(somite_mask, axis=0)
    
    # Create RGB overlay
    rgb = np.zeros((*mip.shape, 3))
    
    # All somites in gray
    rgb[mip > 0] = [0.5, 0.5, 0.5]
    
    colorlist = cc.glasbey_dark
    # Highlight selected somite in a distinct color (e.g., cyan)
    if highlight_tup is not None:
        for i, id in enumerate(highlight_tup):
            color = hex_to_rgb(colorlist[i], normalize=True)
            highlight_mask = (mip == int(id))
            rgb[highlight_mask] = color
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb)    
    plt.show()

def load_somite_roi(nuclear_mask_da, 
                    somite_mask, 
                    df_features, 
                    img_zarr=None,
                    scale_factor=S3_SCALE,
                    pad=50):
    from scipy.ndimage import zoom
    somite_ids = df_features['Somite_ID'].unique().tolist()
    # Calculate bounds from features
    z_min, z_max = int(df_features['Centroid_Z'].min()), int(df_features['Centroid_Z'].max())
    y_min, y_max = int(df_features['Centroid_Y'].min()), int(df_features['Centroid_Y'].max())
    x_min, x_max = int(df_features['Centroid_X'].min()), int(df_features['Centroid_X'].max())
    
    # High-res slices (for nuclear mask and image)
    h_z = slice(z_min - pad, z_max + pad)
    h_y = slice(y_min - pad, y_max + pad)
    h_x = slice(x_min - pad, x_max + pad)
    
    # Low-res slices (somite mask coordinates) via shared resolution conversion.
    l_z = slice(
        int(np.floor(from_s0(z_min - pad, scale_factor))),
        int(np.floor(from_s0(z_max + pad, scale_factor))) + 1,
    )
    l_y = slice(
        int(np.floor(from_s0(y_min - pad, scale_factor))),
        int(np.floor(from_s0(y_max + pad, scale_factor))) + 1,
    )
    l_x = slice(
        int(np.floor(from_s0(x_min - pad, scale_factor))),
        int(np.floor(from_s0(x_max + pad, scale_factor))) + 1,
    )
    
    # print(f"High-res ROI size: {h_z.stop - h_z.start} x {h_y.stop - h_y.start} x {h_x.stop - h_x.start}")
    # print(f"Low-res ROI size: {l_z.stop - l_z.start} x {l_y.stop - l_y.start} x {l_x.stop - l_x.start}")
    print(f"Z slice range: {z_min - pad} - {z_max + pad}")

    somite_roi = np.array(somite_mask[l_z, l_y, l_x])
    somite_upscaled = zoom(somite_roi, scale_factor, order=0)
    
    somite_binary = np.isin(somite_upscaled, somite_ids)
    # somite_binary = (somite_upscaled == int(somite_id))
    roi_shape = (h_z.stop - h_z.start, h_y.stop - h_y.start, h_x.stop - h_x.start)
    
    actual_shape = tuple(min(s1, s2) for s1, s2 in zip(roi_shape, somite_binary.shape))
    somite_binary = somite_binary[:actual_shape[0], :actual_shape[1], :actual_shape[2]]
    
    if img_zarr is not None:
        img_roi = img_zarr[h_z, h_y, h_x][:actual_shape[0], :actual_shape[1], :actual_shape[2]]
    else:
        img_roi = None
    masked_img = np.where(somite_binary, img_roi, 0)
    
    nuclear_roi_da = nuclear_mask_da[h_z, h_y, h_x]
    return {
        'masked_img': masked_img,
        'somite_binary': somite_binary,
        'nuclear_mask_da': nuclear_roi_da,  # Call .compute() when needed
        'z_offset': z_min - pad, # where the ROI starts in full volume -> Use to convert full-volume coords to ROI coords
        'roi_shape': actual_shape,
        'x_offset': x_min - pad,
        'y_offset': y_min - pad
    }

def plot_nuclear_masks(mask_roi,
                       df_features,
                       cluster_col,
                       z_range=None,
                       z_offset=0,
                       show_orthogonal=False,
                       z_factor=0.5,
                       title=None,
                       ax=None,
                       save_dir=None,
                       vmin=None,
                       vmax=None,
                       show_colorbar=True):
    """
    Plot nuclear masks colored by cluster assignment on a pure black background.

    Cells are rendered fully opaque into a single RGB image, so the saved SVG
    does not composite against the viewer's background color.

    Parameters
    ----------
    mask_roi : np.ndarray
        3D nuclear mask array (already computed, not dask)
    df_features : pd.DataFrame
        DataFrame with Nuclear_ID, Centroid_Z, and cluster_col
    cluster_col : str
        Column name for coloring ('cluster_hier' for discrete, else continuous)
    z_range : tuple, optional
        (z_min, z_max) in full volume coordinates. If None, computed from df_features.
    z_offset : int
        Offset to convert full volume coords to ROI coords
    show_orthogonal : bool
        Whether to show XY, XZ, and YZ projections
    z_factor : float
        Fraction of z-span to include (0.5 = full somite, 0.1 = thin slice)
    title : str, optional
        Unused placeholder retained for call-site compatibility
    ax : matplotlib.axes.Axes, optional
        Axis to plot on (ignored if show_orthogonal=True)
    save_dir : str, optional
        Directory to save figure
    vmin, vmax : float, optional
        Fixed color limits for continuous coloring
    show_colorbar : bool
        Draw a colorbar for continuous coloring
    """
    if z_range is None:
        z_min, z_max = df_features['Centroid_Z'].min(), df_features['Centroid_Z'].max()
        z_span = z_max - z_min
        z_center = (z_min + z_max) / 2
        z_range = (z_center - z_factor * z_span, z_center + z_factor * z_span)

    df_slab = df_features[df_features['Centroid_Z'].between(*z_range)].copy()

    z_start = max(0, int(z_range[0] - z_offset))
    z_end = min(mask_roi.shape[0], int(z_range[1] - z_offset))
    mask_slab = mask_roi[z_start:z_end]

    use_discrete = (cluster_col == 'cluster_hier')

    if use_discrete:
        clusters = sorted(df_features[cluster_col].unique())
    else:
        all_values = df_features[cluster_col].dropna().values
        lo = vmin if vmin is not None else all_values.min()
        hi = vmax if vmax is not None else all_values.max()
        norm = Normalize(vmin=lo, vmax=hi)
        cmap = plt.cm.viridis

    def paint_cell(overlay, cell_id, rgb, proj_axis):
        cell_mip = (mask_slab == cell_id).max(axis=proj_axis)
        overlay[cell_mip] = rgb

    projections = [
        (0, 'XY (top-down)'),
        (1, 'XZ (front)'),
        (2, 'YZ (side)'),
    ] if show_orthogonal else [(0, None)]

    if show_orthogonal:
        z_size, y_size, x_size = mask_slab.shape
        fig = plt.figure(figsize=(8, 4 * (y_size + 2 * z_size) / y_size + 1))
        gs = GridSpec(3, 1, height_ratios=[y_size, z_size, z_size], hspace=0.3)
        axes = [fig.add_subplot(gs[i]) for i in range(3)]
    elif ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        axes = [ax]
    else:
        fig = ax.figure
        axes = [ax]

    for ax_i, (proj_axis, proj_title) in zip(axes, projections):
        # Pure black canvas at the projected shape; cells painted opaque on top.
        proj_shape = tuple(s for i, s in enumerate(mask_slab.shape) if i != proj_axis)
        overlay = np.zeros((*proj_shape, 3))

        if use_discrete:
            for c in clusters:
                ids = df_slab.loc[df_slab[cluster_col] == c, 'Nuclear_ID'].values
                rgb = hex_to_rgb(CLUSTER_COLORS[c], normalize=True)
                for cid in ids:
                    paint_cell(overlay, cid, rgb, proj_axis)
        else:
            for _, row in df_slab.iterrows():
                val = row[cluster_col]
                if not np.isnan(val):
                    paint_cell(overlay, row['Nuclear_ID'], cmap(norm(val))[:3], proj_axis)

        # Single opaque draw - no alpha, so nothing blends with the page background.
        ax_i.imshow(np.clip(overlay, 0, 1), origin='lower', aspect='auto',
                    interpolation='nearest')
        if proj_title:
            ax_i.set_title(proj_title, fontsize=11)
        ax_i.axis('off')

        z_text = f"Z: {int(z_range[0])} - {int(z_range[1])}"
        ax_i.text(0.5, -0.05, z_text, transform=ax_i.transAxes,
                  ha='center', va='top', fontsize=10, color='white')

    if use_discrete:
        legend_elements = [Patch(facecolor=CLUSTER_COLORS[c], label=f'Cluster {c}')
                           for c in clusters]
        axes[0].legend(handles=legend_elements, loc='upper left',
                       frameon=False, labelcolor='white')
    elif show_colorbar:
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label(cluster_col, fontsize=10, color='white')
        cbar.ax.tick_params(colors='white')

    if save_dir is not None:
        save_vectorized(save_dir, fig, 'nuclear_masks_mip')

    return fig, axes[0]
    
# def plot_nuclear_masks(mask_roi,
#                        df_features,
#                        cluster_col,
#                        z_range=None,
#                        z_offset=0,
#                        show_orthogonal=False,
#                        z_factor=0.5,
#                        title=None,
#                        ax=None,
#                        save_dir=None,
#                        vmin=None,
#                        vmax=None,
#                        show_colorbar=True):
#     """
#     Plot nuclear masks colored by cluster assignment.
    
#     Parameters:
#     -----------
#     mask_roi : np.ndarray
#         3D nuclear mask array (already computed, not dask)
#     df_features : pd.DataFrame
#         DataFrame with Nuclear_ID, Centroid_Z, and cluster_col
#     cluster_col : str
#         Column name for coloring (e.g., 'cluster_hier' for discrete, others for continuous)
#     z_range : tuple, optional
#         (z_min, z_max) in full volume coordinates. If None, computed from df_features.
#     z_offset : int
#         Offset to convert full volume coords to ROI coords
#     show_orthogonal : bool
#         Whether to show XY, XZ, and YZ projections
#     z_factor : float
#         Fraction of z-span to include (0.5 = full somite, 0.1 = thin slice)
#     title : str, optional
#         Figure title. If None, uses default based on cluster_col.
#     ax : matplotlib.axes.Axes, optional
#         Axis to plot on (ignored if show_orthogonal=True)
#     save_dir : str, optional
#         Directory to save figure
#     """
#     if z_range is None:
#         z_min, z_max = df_features['Centroid_Z'].min(), df_features['Centroid_Z'].max()
#         z_span = z_max - z_min
#         z_center = (z_min + z_max) / 2
#         z_range = (z_center - z_factor * z_span, z_center + z_factor * z_span)

#     df_slab = df_features[df_features['Centroid_Z'].between(*z_range)].copy()

#     z_start = max(0, int(z_range[0] - z_offset))
#     z_end = min(mask_roi.shape[0], int(z_range[1] - z_offset))
#     mask_slab = mask_roi[z_start:z_end]

#     use_discrete = (cluster_col == 'cluster_hier')
    
#     if use_discrete:
#         clusters = sorted(df_features[cluster_col].unique())
#     else:
#         all_values = df_features[cluster_col].dropna().values
#         lo = vmin if vmin is not None else all_values.min()
#         hi = vmax if vmax is not None else all_values.max()
#         norm = Normalize(vmin=lo, vmax=hi)
#         cmap = plt.cm.viridis

#     def paint_cell(overlay, cell_id, rgb, proj_axis):
#         cell_mip = (mask_slab == cell_id).max(axis=proj_axis)
#         overlay[cell_mip] = rgb

#     projections = [
#         (0, 'XY (top-down)'),
#         (1, 'XZ (front)'),
#         (2, 'YZ (side)'),
#     ] if show_orthogonal else [(0, None)]

#     if show_orthogonal:
#         z_size, y_size, x_size = mask_slab.shape
#         fig = plt.figure(figsize=(8, 4 * (y_size + 2 * z_size) / y_size + 1))
#         gs = GridSpec(3, 1, height_ratios=[y_size, z_size, z_size], hspace=0.3)
#         axes = [fig.add_subplot(gs[i]) for i in range(3)]
#     elif ax is None:
#         fig, ax = plt.subplots(1, 1, figsize=(6, 6))
#         axes = [ax]
#     else:
#         fig = ax.figure
#         axes = [ax]

#     for ax_i, (proj_axis, proj_title) in zip(axes, projections):
#         mip_bg = (mask_slab > 0).max(axis=proj_axis).astype(float)
#         overlay = np.zeros((*mip_bg.shape, 3))

#         if use_discrete:
#             for c in clusters:
#                 ids = df_slab.loc[df_slab[cluster_col] == c, 'Nuclear_ID'].values
#                 rgb = hex_to_rgb(CLUSTER_COLORS[c], normalize=True)
#                 for cid in ids:
#                     paint_cell(overlay, cid, rgb, proj_axis)
#         else:
#             for _, row in df_slab.iterrows():
#                 val = row[cluster_col]
#                 if not np.isnan(val):
#                     paint_cell(overlay, row['Nuclear_ID'], cmap(norm(val))[:3], proj_axis)

#         ax_i.imshow(mip_bg, cmap='gray', alpha=0.3, origin='lower', aspect='auto')
#         ax_i.imshow(np.clip(overlay, 0, 1), alpha=0.8, origin='lower', aspect='auto')
#         if proj_title:
#             ax_i.set_title(proj_title, fontsize=11)
#         ax_i.axis('off')
        
#         # Add z-range text below the plot
#         z_text = f"Z: {int(z_range[0])} - {int(z_range[1])}"
#         ax_i.text(0.5, -0.05, z_text, transform=ax_i.transAxes, 
#                   ha='center', va='top', fontsize=10, color='white')
    
#     if use_discrete:
#         legend_elements = [Patch(facecolor=CLUSTER_COLORS[c], label=f'Cluster {c}') for c in clusters]
#         legend = axes[0].legend(handles=legend_elements, loc='upper left', 
#                                 frameon=False, labelcolor='white')
#     elif show_colorbar:
#         sm = ScalarMappable(cmap=cmap, norm=norm)
#         sm.set_array([])
#         cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
#         cbar.set_label(cluster_col, fontsize=10)

#     # fig_title = title if title else f'Cells colored by {cluster_col}'
#     # fig.suptitle(fig_title, fontsize=14)
    
#     if save_dir is not None:
#         save_vectorized(save_dir, fig, 'nuclear_masks_mip')
    
#     return fig, axes[0]


def view_somite_microscopy(result, somite_df):
    somite_id = somite_df['Somite_ID'].values[0]
    z_offset = result['z_offset']
    masked_img = result['masked_img']
    somite_mask = result['somite_binary']
 

    z_min, z_max = somite_df['Centroid_Z'].min(), somite_df['Centroid_Z'].max()
    z_span = z_max - z_min
    z_center = (z_min + z_max) / 2

    # Convert to ROI coordinates by subtracting z_offset
    z_center_roi = z_center - z_offset
    z_span_roi = z_span

    # Define the slab range in ROI coordinates
    z_start = max(0, int(z_center_roi - z_span_roi))
    z_end = min(masked_img.shape[0], int(z_center_roi + z_span_roi))

    print(f"Full volume z range: {z_min:.0f} - {z_max:.0f}")
    print(f"ROI z_offset: {z_offset}")
    print(f"Slab in ROI coords: {z_start} - {z_end}")

    img_slab = masked_img[z_start:z_end]

    plt.figure(figsize=(10, 10))
    plt.imshow(np.max(img_slab, axis=0), cmap='gray')  # axis=0 for z max projection, axis=1 for y
    plt.title(f"Somite {somite_id} - Max Z Projection (z={z_start}-{z_end})")
    plt.colorbar()
    plt.show()

def add_ripleys_k_density(df_features, 
                          n_radii=5, 
                          column_name='ripleys_k_density'):
    """
    Compute local density using Ripley's K function in 3D.
    
    Ripley's K measures the expected number of neighbors within distance r,
    normalized by overall density. Values > 1 indicate local clustering,
    values < 1 indicate local dispersion relative to CSR (complete spatial randomness).
    
    For each point, we compute:
        L(r) = (K(r) / (4/3 * pi))^(1/3) - r
    
    where K(r) = (V/n) * count(neighbors within r), and L(r) is the variance-stabilized
    version. Positive L indicates clustering, negative indicates dispersion.
    
    Parameters
    ----------
    df_features : pd.DataFrame
        DataFrame with 'Centroid_Z', 'Centroid_Y', 'Centroid_X', and 'Nuclear_ID' columns
    radius : float or None
        The search radius for counting neighbors. If None, automatically computed
        as a fraction of the typical inter-point distance.
    n_radii : int
        Number of radii to sample between 0 and max_radius when computing 
        area-under-curve for multi-scale density (default: 5)
    column_name : str
        Name of the column to add to the DataFrame
        
    Returns
    -------
    pd.DataFrame
        Input DataFrame with added '{column_name}' column containing the local
        Ripley's K-based density score.
    
    Notes
    -----
    The returned score is a multi-scale summary: the average L(r) across several
    radii. Higher values indicate stronger local clustering (higher density).
    """
    from scipy.spatial import cKDTree
    
    # 1. Extract 3D coordinates
    z = df_features['Centroid_Z'].values
    y = df_features['Centroid_Y'].values
    x = df_features['Centroid_X'].values
    positions = np.column_stack([z, y, x])
    n = len(positions)
    
    if n < 2:
        df_features = df_features.copy()
        df_features.loc[:, column_name] = np.nan
        return df_features
    
    # 2. Build KD-tree for efficient neighbor queries
    tree = cKDTree(positions)
    
    # 3. Compute bounding box volume for normalization
    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)
    bbox_range = bbox_max - bbox_min
    volume = np.prod(bbox_range) if np.all(bbox_range > 0) else 1.0
    
    # Global intensity (points per unit volume)
    lambda_hat = n / volume
    
    # 4. Determine radius - inter-point spacing as max radius
    # Mean nearest-neighbor distance as baseline
    nn_dists, _ = tree.query(positions, k=2)
    mean_nn_dist = nn_dists[:, 1].mean()
    max_radius = 5 * mean_nn_dist  # cover several neighbor "shells"
    
    # 5. Sample multiple radii for multi-scale analysis
    radii = np.linspace(max_radius / n_radii, max_radius, n_radii)
    
    # 6. Compute local L(r) for each point across multiple radii
    L_scores = np.zeros((n, len(radii)))
    
    for r_idx, r in enumerate(radii):
        # Count neighbors within radius r for each point
        counts = np.array(tree.query_ball_point(positions, r, return_length=True))
        counts = counts - 1  # Exclude self
        
        # Simple edge correction: estimate fraction of sphere within bbox
        # For each point, compute how much of the sphere of radius r
        # is inside the bounding box
        edge_weights = np.ones(n)
        for dim in range(3):
            # Distance to lower and upper boundaries
            d_low = positions[:, dim] - bbox_min[dim]
            d_high = bbox_max[dim] - positions[:, dim]
            
            # Fraction of sphere cut off in this dimension
            # Approximate: if d < r, lose portion proportional to (r-d)/r
            frac_low = np.clip((r - d_low) / (2 * r), 0, 0.5)
            frac_high = np.clip((r - d_high) / (2 * r), 0, 0.5)
            edge_weights *= (1 - frac_low - frac_high)
        
        # Correct counts by edge weight
        edge_weights = np.clip(edge_weights, 0.1, 1.0)  # Avoid division issues
        counts_corrected = counts / edge_weights
        
        # # E(r) = lambda * (4/3) * pi * r^3 (count under CSR)
        # expected_count = lambda_hat * (4/3) * np.pi * (r ** 3)
        
        # K(r) = observed / (n * lambda)
        # local K: K_local(r) = count / (lambda * V_sphere)
        K_local = counts_corrected / (lambda_hat * (4/3) * np.pi * (r ** 3) + 1e-10)
        
        # L(r) = observed/expected - 1 (relative clustering score)
        L_scores[:, r_idx] = K_local - 1  # >0 = clustered, <0 = dispersed
    
    # 7. Aggregate across radii (mean L-function value)
    density_score = L_scores.mean(axis=1)
    # 8. Map to DataFrame
    density_map = dict(zip(df_features['Nuclear_ID'], density_score))
    df_features = df_features.copy()
    df_features.loc[:, column_name] = df_features['Nuclear_ID'].map(density_map)
    return df_features

def add_binary_depth(df_features):
    df_features = df_features.copy()
    df_features.loc[:, 'is_interior'] = 0
    df_features.loc[df_features['depth_normalized'] > 0.5, 'is_interior'] = 1
    return df_features


# def save_vectorized(save_dir, fig, plot_name, transparent=False):
#     from pathlib import Path
#     if save_dir is not None:
#         save_dir = Path(save_dir)
#         if not save_dir.exists():
#             print(f"Creating directory: {save_dir}")
#             save_dir.mkdir(parents=True, exist_ok=True)
#         fig.savefig(
#             save_dir / f"{plot_name}.svg",
#             format="svg",
#             bbox_inches="tight",
#             transparent=transparent,
#             facecolor=None if transparent else 'white',
#             edgecolor='white' if transparent else 'black',
#         )   
#         print(f"Saved plot: {save_dir / f'{plot_name}.svg'}")
#     else:
#         raise ValueError("Directory is required")
#     return


def plot_somite_mip(somite_df, 
                    result, 
                    color_col, 
                    somite_id=None, 
                    show_orthogonal=False, 
                    save_dir=None,
                    ax=None,
                    z_factor=0.5):
    """
    Plot somite visualization using pre-computed result from load_somite_roi.
    
    Prepares the nuclear mask (filtered to somite) and delegates to plot_nuclear_masks.
    
    Parameters:
    -----------
    somite_df : pd.DataFrame
        DataFrame with somite features (Nuclear_ID, Centroid_Z, cluster columns, etc.)
    result : dict
        Output from load_somite_roi containing 'nuclear_mask_da', 'somite_binary', 
        'z_offset', and 'roi_shape'
    color_col : str
        Column name for coloring cells
    somite_id : int, optional
        Somite ID (extracted from somite_df if not provided)
    show_orthogonal : bool
        Whether to show orthogonal projections (XY, XZ, YZ views)
    save_dir : str, optional
        Directory to save figures
    ax : matplotlib.axes.Axes, optional
        Axis to plot on (ignored if show_orthogonal=True)
    z_factor : float
        Fraction of z-span to include (0.5 = full somite)
    """
    if somite_id is None:
        somite_id = int(somite_df['Somite_ID'].values[0])
    
    print("Computing nuclear mask ROI...")
    nuclear_mask_roi = result['nuclear_mask_da'].compute()
    
    actual_shape = result['roi_shape']
    nuclear_mask_roi = nuclear_mask_roi[:actual_shape[0], :actual_shape[1], :actual_shape[2]]
    
    mask_roi_filtered = np.where(result['somite_binary'], nuclear_mask_roi, 0)
    
    somite_side = somite_df['Somite_Side'].values[0]
    somite_stage = somite_df['somite_stage'].values[0]
    if ax is not None:
        ax.set_title(f"ID: {somite_id}, Side: {somite_side}")
        title = ''
    else:
        title = f'Somite ID: {somite_id}, Side: {somite_side}, Stage: {somite_stage}'
    fig, ax_out = plot_nuclear_masks(
        mask_roi_filtered,
        somite_df,
        color_col,
        z_offset=result['z_offset'],
        show_orthogonal=show_orthogonal,
        z_factor=z_factor,
        title=title,
        ax=ax
    )
    return fig, ax_out

def plot_positions_3d(df_features, pos_col):
    import plotly.express as px
    
    fig = px.scatter_3d(
        df_features,
        x="Centroid_X",
        y="Centroid_Y",
        z="Centroid_Z",
        color=pos_col,
        color_continuous_scale="Viridis"
    )
    fig.update_traces(marker=dict(size=2, opacity=0.9))
    fig.update_layout(scene=dict(zaxis=dict(autorange="reversed")))
    fig.show()

def plot_positions_2d(df_features, pos_col):
    import plotly.express as px
    
    fig = px.scatter(
        df_features,
        x="Centroid_X",
        y="Centroid_Y",
        color=pos_col,
        color_continuous_scale="Viridis"
    )
    fig.update_traces(marker=dict(size=5, opacity=0.9))
    fig.show()

def plot_feature_pairwise(df, feature_cols, sample_n=1000, figsize=(12, 12), alpha=0.3):
    """
    Pairwise scatter plots to reveal natural clusters in 2D projections.
    """
    # Subsample for performance
    plot_df = df[feature_cols].dropna()
    if sample_n and len(plot_df) > sample_n:
        plot_df = plot_df.sample(n=sample_n, random_state=42)
    
    n = len(feature_cols)
    fig, axes = plt.subplots(n, n, figsize=figsize)
    
    for i, col_y in enumerate(feature_cols):
        for j, col_x in enumerate(feature_cols):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: histogram
                ax.hist(plot_df[col_x], bins=30, alpha=0.7, color='steelblue', edgecolor='white')
            else:
                # Off-diagonal: scatter
                ax.scatter(plot_df[col_x], plot_df[col_y], alpha=alpha, s=5, c='steelblue')
            
            # Labels only on edges
            if i == n - 1:
                ax.set_xlabel(col_x, fontsize=9)
            else:
                ax.set_xticklabels([])
            
            if j == 0:
                ax.set_ylabel(col_y, fontsize=9)
            else:
                ax.set_yticklabels([])
            
            ax.tick_params(labelsize=7)
    
    plt.suptitle('Pairwise Feature Relationships', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()
    return

def heatmap_features(df_features, feature_cols):
    import seaborn as sns
    
    corr_matrix = df_features[feature_cols].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, square=True)
    plt.title('Feature Correlations')
    plt.tight_layout()
    plt.show()

def print_feature_ranges(df, columns=None):
    """
    Prints a formatted table of min, max, and span for specified columns.
    If columns is None, it uses all numeric columns in the dataframe.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"{'Feature':<20} | {'Min':>10} | {'Max':>10} | {'Span':>10}")
    print("-" * 58)
    for col in columns:
        if col not in df.columns:
            print(f"{col:<20} | {'MISSING':>10} | {'':>10} | {'':>10}")
            continue
        c_min = df[col].min()
        c_max = df[col].max()
        span = c_max - c_min
        # Format for readability: 3 decimal places
        print(f"{col:<20} | {c_min:10.3f} | {c_max:10.3f} | {span:10.3f}")


def plot_feature_distributions_raw(df, bins=30, n_cols=3):
    feature_cols = df.columns
    n_features = len(feature_cols)
    n_rows = int(np.ceil(n_features / n_cols))
    
    figsize = (4 * n_cols, 3 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()
    
    for i, col in enumerate(feature_cols):
        ax = axes[i]
        data = df[col].values
        ax.hist(data, bins=bins, density=True, alpha=0.7, color='steelblue', edgecolor='white')
        # Statistics annotation
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel('Density', fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(labelsize=8)
    plt.suptitle('Raw Feature Distributions', fontsize=14, y=1.02)
    plt.tight_layout()    
    return fig


def pca_features(somite_df, feature_cols, label=None):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    X = somite_df[feature_cols].values
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=len(feature_cols))
    X_pca = pca.fit_transform(X)
    
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )
    importance = loadings.abs().sum(axis=1).sort_values(ascending=False)

    header = f"PCA feature importance ({label})" if label else "PCA feature importance"
    print(header)
    print(importance.to_string())
    
    print(f"\nVariance explained:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var:.1%}")
    print(f"  Total: {pca.explained_variance_ratio_.sum():.1%}")

    return X_pca, pca, loadings

def plot_pca_clusters(X_pca, df, loadings, title=None, save_dir=None, file_name=None):
    clusters = df['cluster_hier'].values
    unique_clusters = sorted(np.unique(clusters))
    
    pc_labels = []
    used_features = set()
    for i in range(2):
        col = loadings.iloc[:, i].abs()
        col_filtered = col[~col.index.str.match(r'^PC\d+$')]
        col_sorted = col_filtered.sort_values(ascending=False)
        for feat in col_sorted.index:
            if feat not in used_features:
                top_feature = feat
                used_features.add(feat)
                break
        pc_labels.append(f'PC{i+1}')
    
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    
    for c in unique_clusters:
        mask = clusters == c
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],  
                   c=CLUSTER_COLORS[c], label=f'Cluster {c}', 
                   alpha=0.9, s=20, edgecolors='none')
    
    ax.set_xlabel(pc_labels[0])
    ax.set_ylabel(pc_labels[1])
    ax.set_title(title or 'PCA of Feature Space Colored by Hierarchical Clustering')
    ax.legend()
    plt.tight_layout()

    if save_dir and file_name is not None:
        save_vectorized(save_dir, fig, file_name)
    return fig


def plot_class_distributions(df, cluster_col):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    df = df.copy()
    df = df[df[cluster_col].notna()]
    sns.histplot(df[cluster_col], bins=30, kde=True, color=CLUSTER_COLORS[cluster_col])
    plt.show()
    return

# def plot_feature_distribution_by_cluster(df, feature_col, ax):
#     df = df.copy()
#     df = df[df[feature_col].notna()]
#     classes = df['cluster_hier'].unique()
#     for c in classes:
#         mask = df['cluster_hier'] == c
#         sns.histplot(df[feature_col].values[mask], bins=30, kde=False, color=CLUSTER_COLORS[c], ax=ax)
#     ax.set_xlabel(feature_col)
#     ax.set_ylabel('Count')
#     return ax

# fig1, axes1 = plt.subplots(1, len(feature_cols), figsize=(20, 5))
# fig2, axes2 = plt.subplots(1, len(feature_cols), figsize=(20, 5))

# for i, feature_col in enumerate(feature_cols):
#     left_df = full_df[full_df['Somite_Side'] == 'Left']
#     right_df = full_df[full_df['Somite_Side'] == 'Right']
#     plot_feature_distribution_by_cluster(left_df, feature_col, axes1[i])
#     plot_feature_distribution_by_cluster(right_df, feature_col, axes2[i])
# plt.show()


# =============================================================================
# feature_extract figures
# =============================================================================

def plot_position_scatter(positions_df, save_dir, embryo_id,
                          camera_angles=None, camera_center=None,
                          camera_zoom=None, camera_s_scale=4.0,
                          n_sample=70_000, seed=42):
    """Save a Scatter3d plot for each biological-position column.

    When *camera_angles*, *camera_center*, and *camera_zoom* are provided the
    figures use a fixed Napari-derived camera so every run produces the same
    viewpoint.
    """
    import plotly.graph_objects as go

    n = min(n_sample, len(positions_df))
    sample = positions_df.sample(n=n, seed=seed)

    x_vals = sample['Centroid_X'].to_numpy()
    y_vals = sample['Centroid_Y'].to_numpy()
    z_vals = sample['Centroid_Z'].to_numpy()

    camera = None
    if camera_angles and camera_center and camera_zoom:
        camera = napari_camera_to_plotly(
            angles=camera_angles,
            center=camera_center,
            zoom=camera_zoom,
            s_scale=camera_s_scale,
            x_vals=x_vals, y_vals=y_vals, z_vals=z_vals,
        )

    fig_dir = Path(save_dir) / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    for col in ('pos_AP', 'pos_LR', 'pos_DV'):
        color_vals = sample[col].to_numpy()
        fig = go.Figure(data=go.Scatter3d(
            x=x_vals, y=y_vals, z=z_vals,
            mode='markers',
            marker=dict(
                size=3, color=color_vals, opacity=1,
                colorbar=dict(title=col, len=0.5, x=0.8, y=0.5),
                cmid=0 if col == 'pos_LR' else None,
            ),
        ))
        layout_kwargs = dict(
            title=None,
            scene=dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                aspectmode='data',
            ),
            height=800,
        )
        if camera is not None:
            layout_kwargs['scene_camera'] = camera
        fig.update_layout(**layout_kwargs)

        out_path = fig_dir / f'{embryo_id}_{col}.png'
        fig.write_image(str(out_path), width=1000, height=1000, scale=3)
        logger.info(f"Saved position scatter plot to {out_path}")


def plot_alignment_mips(somite_df, nuclear_mask, somite_mask, save_dir,
                        stage=13, z_factor=0.05,
                        eccentricity_min=0.3):
    """Render the L+R somite pair at one stage as 3 MIP panels colored by
    align_AP, align_DV, align_LR.

    All three panels share a fixed colormap range [0, 1] so they are directly
    comparable. A thin Z slab (controlled by ``z_factor``) is used to avoid
    collapsing the full stage span into a single projection. Nuclei with
    ``eccentricity < eccentricity_min`` are dropped before plotting, since
    the major-axis direction of a near-spherical nucleus is dominated by
    eigenvector noise.
    """
    if hasattr(somite_df, "to_pandas"):
        somite_df = somite_df.to_pandas()

    align_cols = ["align_AP", "align_DV", "align_LR"]
    missing = [c for c in align_cols if c not in somite_df.columns]
    if missing:
        logger.warning(f"Missing alignment columns {missing}; skipping alignment MIP")
        return

    sub_df = somite_df[somite_df["Somite_Stage"] == stage].copy()
    if sub_df.empty:
        logger.warning(f"No nuclei found for Somite_Stage={stage}; skipping alignment MIP")
        return

    if "eccentricity" in sub_df.columns:
        n_before = len(sub_df)
        sub_df = sub_df[sub_df["eccentricity"] >= eccentricity_min].copy()
        logger.info(
            f"Eccentricity filter ≥ {eccentricity_min}: "
            f"{len(sub_df)}/{n_before} nuclei retained"
        )
        if sub_df.empty:
            logger.warning(
                f"No nuclei pass eccentricity ≥ {eccentricity_min} at stage {stage}"
            )
            return
    else:
        logger.warning(
            "No 'eccentricity' column; plotting all nuclei (including round ones)"
        )

    n_left = (sub_df["Side"].str.lower() == "left").sum()
    n_right = (sub_df["Side"].str.lower() == "right").sum()
    logger.info(f"Rendering alignment MIPs for stage {stage} "
                f"({n_left} left + {n_right} right nuclei)")

    result = load_somite_roi(
        nuclear_mask_da=nuclear_mask,
        somite_mask=somite_mask,
        df_features=sub_df,
    )

    nuclear_mask_roi = result["nuclear_mask_da"].compute()
    actual = result["roi_shape"]
    nuclear_mask_roi = nuclear_mask_roi[:actual[0], :actual[1], :actual[2]]
    mask_roi_filtered = np.where(result["somite_binary"], nuclear_mask_roi, 0)

    z_min, z_max = sub_df["Centroid_Z"].min(), sub_df["Centroid_Z"].max()
    z_center = (z_min + z_max) / 2
    z_half = z_factor * (z_max - z_min)
    z_range = (z_center - z_half, z_center + z_half)
    logger.info(
        f"Z slab: {int(z_range[0])}–{int(z_range[1])} "
        f"({int(2 * z_half)} slices, center {int(z_center)})"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col in zip(axes, align_cols):
        plot_nuclear_masks(
            mask_roi_filtered,
            sub_df,
            col,
            z_range=z_range,
            z_offset=result["z_offset"],
            ax=ax,
            vmin=0.0,
            vmax=1.0,
            show_colorbar=False,
        )
        ax.set_title(col, color="white", fontsize=11)

    sm = ScalarMappable(cmap=plt.cm.viridis, norm=Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical",
                        fraction=0.02, pad=0.04)
    cbar.set_label("|direction cosine|", fontsize=10, color="white")
    cbar.ax.tick_params(colors="white")

    fig.suptitle(f"Stage {stage} — somite alignment", color="white")
    save_vectorized(save_dir, fig, f"stage_{stage}_alignment_mip", transparent=True)
    plt.close(fig)
    logger.info(f"Saved alignment MIP for stage {stage}")
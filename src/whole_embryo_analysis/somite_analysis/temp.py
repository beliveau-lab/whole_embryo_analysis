"""
Hybrid Anatomical Frame

Combines:
    - T (AP axis): From Frenet-Serret frame of the neural tube spline
      (more accurate than somite-midline tangent for following true AP curvature)
    - LR axis: From paired L/R somite vectors (biologically grounded)
    - DV axis: T × LR (right-hand rule)

This is the recommended frame for chirality analysis because:
    1. AP follows the neural tube faithfully (better than somite midline)
    2. LR is defined by real anatomy (unlike Frenet N/B)
    3. The mirror plane (LR=0) is unambiguous for the reflection operator

Add this function to anatomical_coordinates.py alongside compute_anatomical_frame
and compute_frenet_frame.
"""

import numpy as np
import polars as pl
import logging
from scipy.interpolate import splprep, splev, interp1d
from scipy.spatial import cKDTree
from typing import Tuple

# These are already in your anatomical_coordinates.py
from anatomical_coordinates import (
    AnatomicalFrame,
    FrenetFrame,
    get_somite_centroids,
    calculate_frenet_frames,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Add to anatomical_coordinates.py
# =============================================================================

def compute_hybrid_frame(somite_lut: pl.DataFrame,
                         spline_points: np.ndarray,
                         pair_index_col: str = 'somite_stage',
                         side_col: str = 'somite_side',
                         spline_smoothing: float = 2.0,
                         midline_smoothing: float = 5.0,
                         spline_scale_factor: float = 8.0,
                         u_resolution: int = 1000) -> AnatomicalFrame:
    """
    Build hybrid coordinate system: neural tube spline for AP + somite pairs for LR.
    - The neural tube spline provides a smooth, anatomically faithful AP axis
        that follows the true anterior-posterior curvature of the embryo.
    - Paired L/R somites define the left-right axis biologically

    Pipeline:
        1. Fit spline to neural tube points → compute Frenet tangent T at each u
        2. Fit spline to L/R somite midpoints → this becomes the "query curve"
           (we need it to know where along the neural tube each somite pair lives)
        3. For each somite pair, find the closest point on the neural tube spline
        4. Interpolate LR vectors along the neural tube parameterization
        5. Orthogonalize: LR_perp = LR - (LR · T)T, then DV = T × LR_perp
    
    Args:
        somite_lut: DataFrame with centroid_z/y/x columns and somite metadata.
                    Must already have centroids computed (call get_somite_centroids first).
        spline_points: (N, 3) array of neural tube control points in ZYX order,
                       in the SAME coordinate space as somite centroids.
        pair_index_col: Column identifying matched L/R pairs (e.g., 'somite_stage')
        side_col: Column identifying left vs right ('somite_side')
        spline_smoothing: Smoothing parameter for the neural tube spline
        midline_smoothing: Smoothing parameter for the somite midline spline
                          (only used for the output AnatomicalFrame's tck, not for T)
        spline_scale_factor: Scale factor applied to spline_points to match
                             somite centroid coordinates
        u_resolution: Number of evaluation points along the curve
        
    Returns:
        AnatomicalFrame with T from neural tube, LR from somite pairs, DV = T × LR
    """
    
    # ------------------------------------------------------------------
    # 1. Build neural tube spline and compute Frenet tangents
    # ------------------------------------------------------------------
    spline_pts = np.asarray(spline_points) * spline_scale_factor
    
    if spline_pts.shape[1] != 3:
        raise ValueError(f"spline_points must be (N, 3), got {spline_pts.shape}")
    if len(spline_pts) < 2:
        raise ValueError("Need at least 2 spline points for the neural tube")
    
    k_nt = min(3, len(spline_pts) - 1)
    tck_nt, _ = splprep(spline_pts.T, s=spline_smoothing, k=k_nt)
    
    logger.info(f"Neural tube spline: {len(spline_pts)} control points, k={k_nt}")
    
    # ------------------------------------------------------------------
    # 2. Extract paired L/R somite midpoints and LR vectors
    # ------------------------------------------------------------------
    left_somites = somite_lut.filter(
        pl.col(side_col).str.to_lowercase().str.starts_with('l')
    ).sort(pair_index_col)
    
    right_somites = somite_lut.filter(
        pl.col(side_col).str.to_lowercase().str.starts_with('r')
    ).sort(pair_index_col)
    
    def build_coord_dict(df):
        return {
            int(float(row[pair_index_col])): np.array([
                row['centroid_z'], row['centroid_y'], row['centroid_x']
            ])
            for row in df.iter_rows(named=True)
        }
    
    left_dict = build_coord_dict(left_somites)
    right_dict = build_coord_dict(right_somites)
    paired_numbers = sorted(set(left_dict.keys()) & set(right_dict.keys()))
    
    if len(paired_numbers) < 2:
        raise ValueError(
            f"Need at least 2 paired L/R somites, found {len(paired_numbers)}. "
            f"Left stages: {sorted(left_dict.keys())}, Right stages: {sorted(right_dict.keys())}"
        )
    
    logger.info(f"Found {len(paired_numbers)} paired somites: {paired_numbers}")
    
    midpoints = []
    lr_vectors = []
    for num in paired_numbers:
        L = left_dict[num]
        R = right_dict[num]
        midpoints.append((L + R) / 2)
        lr_vectors.append(R - L)  # Points from Left toward Right
    
    midpoints = np.array(midpoints)   # (N_pairs, 3) in ZYX
    lr_vectors = np.array(lr_vectors)  # (N_pairs, 3) in ZYX
    
    # ------------------------------------------------------------------
    # 3. Map somite midpoints onto neural tube parameterization
    #    Find the u parameter on the neural tube spline closest to each midpoint
    # ------------------------------------------------------------------
    from scipy.optimize import minimize_scalar
    
    # Build a KD-tree on a dense sampling of the neural tube spline
    u_nt_dense = np.linspace(0, 1, u_resolution)
    curve_nt_dense = np.array(splev(u_nt_dense, tck_nt)).T
    kdt_nt = cKDTree(curve_nt_dense)
    
    u_somites_on_nt = np.zeros(len(paired_numbers))
    for i, mp in enumerate(midpoints):
        _, idx = kdt_nt.query(mp)
        u_seed = u_nt_dense[idx]
        bracket = (max(0, u_seed - 0.05), min(1, u_seed + 0.05))
        
        res = minimize_scalar(
            lambda u: np.sum((np.array(splev(u, tck_nt)) - mp) ** 2),
            bounds=bracket, method='bounded',
            options={'xatol': 1e-8}
        )
        u_somites_on_nt[i] = res.x if res.success else u_seed
    
    logger.info(f"Somite u-params on neural tube: "
                f"[{u_somites_on_nt.min():.3f}, {u_somites_on_nt.max():.3f}]")
    
    # ------------------------------------------------------------------
    # 4. Compute Frenet tangent T along the neural tube spline
    # ------------------------------------------------------------------
    T_dense, _, _, valid_T, _, _ = calculate_frenet_frames(u_nt_dense, tck_nt)
    
    if not valid_T.all():
        n_invalid = (~valid_T).sum()
        logger.warning(f"{n_invalid} points have undefined tangent on neural tube spline")
    
    # ------------------------------------------------------------------
    # 5. Interpolate LR vectors along the neural tube parameterization
    # ------------------------------------------------------------------
    lr_norms = np.linalg.norm(lr_vectors, axis=1, keepdims=True)
    lr_unit = lr_vectors / lr_norms
    
    # Ensure consistent LR orientation along the curve
    # (check that consecutive LR vectors don't flip)
    for i in range(1, len(lr_unit)):
        if np.dot(lr_unit[i], lr_unit[i - 1]) < 0:
            logger.warning(
                f"LR vector flips between somite stages {paired_numbers[i-1]} "
                f"and {paired_numbers[i]}. Check somite labels."
            )
    
    f_lr = interp1d(
        u_somites_on_nt, lr_unit, axis=0, kind='linear',
        bounds_error=False, fill_value=(lr_unit[0], lr_unit[-1])
    )
    lr_dense = f_lr(u_nt_dense)  # (u_resolution, 3)
    
    # ------------------------------------------------------------------
    # 6. Orthogonalize: remove T component from LR, compute DV = T × LR
    # ------------------------------------------------------------------
    # LR_perp = LR - (LR · T) * T
    dot_product = np.sum(lr_dense * T_dense, axis=1, keepdims=True)
    lr_perp = lr_dense - dot_product * T_dense
    lr_perp_norms = np.linalg.norm(lr_perp, axis=1, keepdims=True)
    
    # Check for degenerate cases (LR nearly parallel to T)
    degenerate = lr_perp_norms.ravel() < 1e-6
    if degenerate.any():
        logger.warning(
            f"{degenerate.sum()} points have LR nearly parallel to T. "
            f"These will use un-orthogonalized LR."
        )
        lr_perp[degenerate] = lr_dense[degenerate]
        lr_perp_norms[degenerate] = np.linalg.norm(
            lr_perp[degenerate], axis=1, keepdims=True
        )
    
    lr_perp /= lr_perp_norms
    
    # DV = T × LR (right-hand rule)
    dv_dense = np.cross(T_dense, lr_perp)
    dv_norms = np.linalg.norm(dv_dense, axis=1, keepdims=True)
    dv_dense /= np.maximum(dv_norms, 1e-9)
    
    # ------------------------------------------------------------------
    # 7. Also fit a midline spline through somite midpoints
    #    (stored in AnatomicalFrame.tck for projection queries)
    #    We use the NEURAL TUBE spline as the primary curve, but store
    #    somite midpoints for reference.
    # ------------------------------------------------------------------
    
    logger.info(
        f"Hybrid frame built: neural tube T + {len(paired_numbers)} somite pairs for LR. "
        f"Max |LR·T| before orthogonalization: {np.abs(dot_product).max():.4f}"
    )
    
    return AnatomicalFrame(
        tck=tck_nt,           # Neural tube spline (used for projection queries)
        u_dense=u_nt_dense,
        curve_dense=curve_nt_dense,
        T=T_dense,            # From neural tube Frenet tangent
        LR=lr_perp,           # From somite pairs, orthogonalized to T
        DV=dv_dense,          # T × LR
        kdt=kdt_nt,           # KD-tree on neural tube curve
        paired_numbers=paired_numbers,
        u_somites=u_somites_on_nt,
        midpoints=midpoints
    )


# =============================================================================
# Add to somite_analysis.py
# =============================================================================

def build_hybrid_frame(somite_mask: np.ndarray,
                       somite_lut: pl.DataFrame,
                       spline_csv_path: str,
                       mask_label_col: str = 'Somite_ID',
                       pair_index_col: str = 'somite_stage',
                       side_col: str = 'somite_side',
                       spline_smoothing: float = 2.0,
                       midline_smoothing: float = 5.0,
                       somite_scale_factor: float = 8.0,
                       spline_scale_factor: float = 8.0,
                       spline_axis_cols: Tuple[str, str, str] = ('axis-0', 'axis-1', 'axis-2')
                       ) -> AnatomicalFrame:
    """
    Build hybrid coordinate system from neural tube spline + somite pairs.
    
    This is the RECOMMENDED frame for chirality analysis because it gives:
        - Accurate AP from neural tube geometry
        - Biologically defined LR from actual somite pair vectors
        - Unambiguous mirror plane for the reflection operator
    
    Args:
        somite_mask: 3D labeled somite mask (ZYX)
        somite_lut: DataFrame with somite metadata
        spline_csv_path: Path to CSV with neural tube spline points
        mask_label_col: Column in LUT with mask label IDs
        pair_index_col: Column identifying L/R pair matches
        side_col: Column identifying left vs right
        spline_smoothing: Smoothing for neural tube spline
        midline_smoothing: Smoothing for somite midline (stored in output)
        somite_scale_factor: Scale factor for somite mask centroids
        spline_scale_factor: Scale factor for spline point coordinates
        spline_axis_cols: Column names in spline CSV (z, y, x order)
        
    Returns:
        AnatomicalFrame with hybrid T/LR/DV
    """
    import pandas as pd
    
    # Compute somite centroids
    logger.info(f"Computing somite centroids (scale={somite_scale_factor})...")
    somite_lut = get_somite_centroids(
        somite_mask, somite_lut,
        mask_label_col=mask_label_col,
        scale_factor=somite_scale_factor
    )
    
    # Load neural tube spline points
    logger.info(f"Loading neural tube spline from {spline_csv_path}...")
    spline_df = pd.read_csv(spline_csv_path)
    spline_points = np.column_stack([
        spline_df[spline_axis_cols[0]].values,
        spline_df[spline_axis_cols[1]].values,
        spline_df[spline_axis_cols[2]].values
    ])
    
    logger.info(f"Neural tube: {len(spline_points)} control points")
    
    # Build hybrid frame
    frame = compute_hybrid_frame(
        somite_lut=somite_lut,
        spline_points=spline_points,
        pair_index_col=pair_index_col,
        side_col=side_col,
        spline_smoothing=spline_smoothing,
        midline_smoothing=midline_smoothing,
        spline_scale_factor=spline_scale_factor,
    )
    
    logger.info(f"Hybrid frame built: {len(frame.paired_numbers)} somite pairs")
    return frame
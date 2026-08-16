import os
import random 
import numpy as np
import pandas as pd
import zarr
import dask.array as da
import pyvista as pv
from skimage import measure
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import polars as pl
from colorcet import glasbey_dark
import logging

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import polars as pl
from utils import madi_utils
import tifffile
from tqdm import tqdm
from scipy.interpolate import splprep, splev, interp1d
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree

def get_somite_centroids(somite_mask, somite_lut, label_col='Somite_ID'):
    regions = measure.regionprops(somite_mask)
    labels = []
    centroids_z = []
    centroids_y = []
    centroids_x = []
    
    for prop in regions:
        sid = prop.label
        labels.append(sid)
        cz, cy, cx = prop.centroid
        centroids_z.append(cz)
        centroids_y.append(cy)
        centroids_x.append(cx)
    
    centroids_df = pl.DataFrame({
        label_col: labels,
        'centroid_z': centroids_z,
        'centroid_y': centroids_y,
        'centroid_x': centroids_x
    })
    somite_lut = somite_lut.join(centroids_df, on=label_col, how='inner')
    return somite_lut

def calculate_bishop_frames(u, tck, u_res=1000):
    """
    Compute Parallel Transport (Bishop) frames along the spline.
    This prevents the flipping artifacts seen in Frenet-Serret frames.
    u: scalar or array of parameters in [0, 1] to evaluate
    tck: spline representation from splprep
    u_res: resolution of the dense path for propagation
    Returns: T, N, B (normalized basis vectors)
    """
    u_eval = np.atleast_1d(u)
    
    # 1. Generate a dense path for propagation to minimize discretization error
    u_dense = np.linspace(0, 1, u_res)
    points_dense = np.array(splev(u_dense, tck)).T
    tangents_all = np.array(splev(u_dense, tck, der=1)).T
    tangents_all /= np.linalg.norm(tangents_all, axis=1, keepdims=True)
    
    # 2. Define initial frame at u=0
    T0 = tangents_all[0]
    # Pick an arbitrary vector not parallel to T0 to define initial Normal
    v_ref = np.array([0, 0, 1]) if abs(T0[2]) < 0.9 else np.array([0, 1, 0])
    B0 = np.cross(T0, v_ref)
    B0 /= np.linalg.norm(B0)
    N0 = np.cross(B0, T0)
    
    # 3. Propagate frame using Double Reflection method (stable parallel transport)
    # References: "Computation of Rotation Minimizing Frames" by Wang et al.
    T = tangents_all
    N = np.zeros_like(T)
    B = np.zeros_like(T)
    N[0], B[0] = N0, B0
    
    for i in range(len(u_dense) - 1):
        # Double reflection to move from T[i] to T[i+1]
        v1 = points_dense[i+1] - points_dense[i]
        c1 = np.dot(v1, v1)
        if c1 < 1e-12:
            N[i+1], B[i+1] = N[i], B[i]
            continue
            
        # Reflection 1
        n_tilde = N[i] - (2.0 / c1) * np.dot(v1, N[i]) * v1
        t_tilde = T[i] - (2.0 / c1) * np.dot(v1, T[i]) * v1
        
        # Reflection 2
        v2 = T[i+1] - t_tilde
        c2 = np.dot(v2, v2)
        if c2 < 1e-12:
            N[i+1] = n_tilde
        else:
            N[i+1] = n_tilde - (2.0 / c2) * np.dot(v2, n_tilde) * v2
            
        N[i+1] /= np.linalg.norm(N[i+1])
        B[i+1] = np.cross(T[i+1], N[i+1])

    # 4. Interpolate frames to the requested u values
    f_N = interp1d(u_dense, N, axis=0, kind='linear', fill_value="extrapolate")
    f_B = interp1d(u_dense, B, axis=0, kind='linear', fill_value="extrapolate")
    f_T = interp1d(u_dense, T, axis=0, kind='linear', fill_value="extrapolate")
    
    T_out = f_T(u_eval)
    N_out = f_N(u_eval)
    B_out = f_B(u_eval)
    
    # Final normalization
    T_out /= np.linalg.norm(T_out, axis=1, keepdims=True)
    N_out /= np.linalg.norm(N_out, axis=1, keepdims=True)
    B_out /= np.linalg.norm(B_out, axis=1, keepdims=True)
    
    if np.isscalar(u) or (u_eval.ndim == 1 and u_eval.shape[0] == 1 and not isinstance(u, (np.ndarray, list))):
        return T_out.squeeze(), N_out.squeeze(), B_out.squeeze()
    return T_out, N_out, B_out

def calculate_frenet_frames(u, tck):
    """
    Compute robust Frenet-Serret frame [T, N, B] at parameter u.
    u: scalar or array of parameters in [0, 1]
    tck: spline representation from splprep
    Returns: T, N, B (normalized basis vectors)
    """
    u = np.asarray(u)
    is_scalar = (u.ndim == 0)
    if is_scalar:
        u_arr = u[np.newaxis]
    else:
        u_arr = u

    # 1st and 2nd derivatives
    d1 = np.array(splev(u_arr, tck, der=1)).T
    d2 = np.array(splev(u_arr, tck, der=2)).T

    if d1.ndim == 1: d1 = d1[np.newaxis, :]
    if d2.ndim == 1: d2 = d2[np.newaxis, :]

    # Tangent T
    norms_d1 = np.linalg.norm(d1, axis=1, keepdims=True)
    T = np.zeros_like(d1)
    valid_t = (norms_d1 > 1e-9).squeeze()
    T[valid_t] = d1[valid_t] / norms_d1[valid_t]

    # Binormal B = (T x d2) / |T x d2|
    B_cand = np.cross(T, d2, axis=1)
    norms_B = np.linalg.norm(B_cand, axis=1, keepdims=True)
    B = np.zeros_like(B_cand)
    valid_b = (norms_B > 1e-9).squeeze()
    B[valid_b] = B_cand[valid_b] / norms_B[valid_b]

    # Fallback for straight sections
    invalid_b = ~valid_b
    if np.any(invalid_b):
        for v in [np.array([0, 1, 0]), np.array([1, 0, 0]), np.array([0, 0, 1])]:
            B_cand_alt = np.cross(T[invalid_b], v)
            norms_alt = np.linalg.norm(B_cand_alt, axis=1, keepdims=True)
            valid_alt = (norms_alt > 1e-9).squeeze()
            if np.any(valid_alt):
                # Work with indices since B[invalid_b] might be a view or copy
                idx_invalid = np.where(invalid_b)[0][valid_alt]
                B[idx_invalid] = B_cand_alt[valid_alt] / norms_alt[valid_alt]
                break

    # Normal N = B x T
    N = np.cross(B, T, axis=1)
    norms_N = np.linalg.norm(N, axis=1, keepdims=True)
    valid_n = (norms_N > 1e-9).squeeze()
    N[valid_n] = N[valid_n] / norms_N[valid_n]

    if is_scalar:
        return T.squeeze(), N.squeeze(), B.squeeze()
    return T, N, B

def _distance_sq(u, p, tck):
    s_u = np.array(splev(u, tck))
    return np.sum((s_u - p)**2)

def project_points_to_spline(points, tck, u_res=1001):
    """
    Project points onto spline to find u and biological coordinates.
    Follows robust approach from axis_transform.py.
    points: (N, 3) in ZYX
    tck: spline representation
    Returns: pos_AP (u), pos_DV (Normal), pos_LR (Binormal)
    """
    points = np.atleast_2d(points)
    
    # Dense sampling for KD-tree seeding
    u_dense = np.linspace(0, 1, u_res)
    curve_dense = np.array(splev(u_dense, tck)).T
    kdt = cKDTree(curve_dense)
    
    pos_ap = np.zeros(len(points))
    pos_dv = np.zeros(len(points))
    pos_lr = np.zeros(len(points))
    
    for i, p in enumerate(points):
        # 1. Seed from KD-tree
        dist, idx = kdt.query(p)
        u_seed = u_dense[idx]
        
        # 2. Refine with local minimization
        bracket = (max(0, u_seed - 0.05), min(1, u_seed + 0.05))
        res = minimize_scalar(_distance_sq, bounds=bracket, args=(p, tck), method='bounded')
        u_opt = res.x if res.success else u_seed
        
        # 3. Calculate basis and coordinates
        p_proj = np.array(splev(u_opt, tck))
        T, N, B = calculate_bishop_frames(u_opt, tck)
        
        V = p - p_proj
        pos_ap[i] = u_opt
        pos_dv[i] = np.dot(V, N)
        pos_lr[i] = np.dot(V, B)
        
    return pos_ap, pos_dv, pos_lr

def compute_somite_anatomy(somite_lut, label_col='Somite_ID', s=5):
    """
    Fits spline to somite centroids and computes bases/coords using Bishop frames.
    Increased default smoothing (s=5) for better stability.
    Returns:
        somite_lut: updated with pos_AP, pos_DV, pos_LR
        frames: dict of {somite_id: R} where R = [T, N, B] matrix
        tck: the fitted spline (to be used for nuclei projection)
    """
    sorted_lut = somite_lut.sort(label_col)
    sids = sorted_lut[label_col].to_list()
    centroids = sorted_lut.select(['centroid_z', 'centroid_y', 'centroid_x']).to_numpy()
    
    # Fit spline through centroids (ONLY ONCE)
    tck, u = splprep(centroids.T, s=s, k=min(3, len(centroids)-1))
    
    # Compute all Bishop frames for the somite centroids
    T_all, N_all, B_all = calculate_bishop_frames(u, tck)
    
    frames = {}
    pos_dv = []
    pos_lr = []
    
    for i, sid in enumerate(sids):
        u_val = u[i]
        p_orig = centroids[i]
        p_proj = np.array(splev(u_val, tck))
        
        T, N, B = T_all[i], N_all[i], B_all[i]
        frames[sid] = np.column_stack([T, N, B])
        
        V = p_orig - p_proj
        pos_dv.append(np.dot(V, N))
        pos_lr.append(np.dot(V, B))
        
    coord_df = pl.DataFrame({
        label_col: sids,
        'pos_AP': u,
        'pos_DV': pos_dv,
        'pos_LR': pos_lr
    })
    
    updated_lut = somite_lut.join(coord_df, on=label_col, how='left')
    return updated_lut, frames, tck

def compute_frenet_frames(somite_lut, label_col='Somite_ID'):
    """Legacy compatibility for somite_analysis.py - now uses Bishop frames"""
    _, frames, _ = compute_somite_anatomy(somite_lut, label_col=label_col)
    return frames

def plot_somite_bases(somite_lut, somite_bases, scale=10):
    """Plot somite bases using precomputed centroids."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['red', 'green', 'blue']  # PC1, PC2, PC3
    labels = ['PC1 (AP)', 'PC2 (DV)', 'PC3 (LR)']
    # Convert to dict for fast lookup
    centroid_lookup = {
        row['Somite_ID']: (row['centroid_z'], row['centroid_y'], row['centroid_x'])
        for row in somite_lut.iter_rows(named=True)
    }
    
    for sid, R in somite_bases.items():
        if sid not in centroid_lookup:
            continue
        z, y, x = centroid_lookup[sid]
        
        for i in range(3):
            dz, dy, dx = R[:, i] * scale
            ax.quiver(x, y, z, dx, dy, dz, color=colors[i], 
                      arrow_length_ratio=0.1, linewidth=2)
    # Add legend
    for i, (c, l) in enumerate(zip(colors, labels)):
        ax.quiver([], [], [], [], [], [], color=c, label=l)
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Somite Local Coordinate Frames')
    plt.tight_layout()
    plt.show()
    return fig


def main():
    somite_mask_path = '/net/beliveau/vol1/project/EKN_whole_embryo_structural_analysis/Somites/E9.5_306/306_all_somites_no_PSM.tif'
    somite_mask = tifffile.imread(somite_mask_path)
    somite_lut = pl.read_csv('/net/beliveau/vol2/instrument/E9.5_306/Zoom_306/306_somite_lookup_table.csv')

    somite_lut = get_somite_centroids(somite_mask, somite_lut)
    somite_lut, somite_bases, tck = compute_somite_anatomy(somite_lut)

blob:vscode-webview://0v9m7u11o65unhe9ha0i9di3b0v3kagpqsh0p06kflc3o6pnks69/222c5a52-a85b-4feb-aaef-eb54cb4b219b    print(somite_lut.head())
    
    plot_somite_bases(somite_lut, somite_bases)
    plt.show()

if __name__ == "__main__":
    main()

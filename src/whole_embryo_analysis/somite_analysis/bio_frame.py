"""
Anatomical Coordinate System for Mouse Embryo Analysis

Computes a global anatomical coordinate system (AP, LR, DV) from paired
left/right somites and projects nuclear centroids and orientations into it.

All coordinates operate in s0 (original resolution) space. Conversions from
mask (s3) and spline (s4) spaces are handled via resolution_conversion.

Three coordinate frame approaches:
    1. AnatomicalFrame: L/R somite pairs define midline + LR axis
    2. FrenetFrame: Pure curve geometry from neural tube spline
    3. Hybrid (recommended): Neural tube spline for AP + somite pairs for LR
"""

import numpy as np
import polars as pl
from scipy.interpolate import splprep, splev, interp1d
from scipy.spatial import cKDTree
from scipy.optimize import minimize_scalar
from skimage import measure
from dataclasses import dataclass
import logging

from whole_embryo_analysis.utils.scale_resolution import to_s0, from_s0, S3_SCALE, S4_SCALE

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AnatomicalFrame:
    tck: tuple
    u_dense: np.ndarray
    curve_dense: np.ndarray
    T: np.ndarray
    LR: np.ndarray
    DV: np.ndarray
    kdt: cKDTree
    paired_numbers: list
    u_somites: np.ndarray
    midpoints: np.ndarray


@dataclass
class FrenetFrame:
    tck: tuple
    u_dense: np.ndarray
    curve_dense: np.ndarray
    T: np.ndarray
    N: np.ndarray
    B: np.ndarray
    valid_T: np.ndarray
    valid_N: np.ndarray
    valid_B: np.ndarray
    kdt: cKDTree


# =============================================================================
# Helpers
# =============================================================================

def _distance_sq(u, point, tck):
    s_u = np.array(splev(u, tck))
    return np.sum((s_u - point)**2)


def _build_coord_dict(df, pair_index_col):
    return {
        int(float(row[pair_index_col])): np.array([
            row['Centroid_Z'], row['Centroid_Y'], row['Centroid_X']
        ])
        for row in df.iter_rows(named=True)
    }


def _extract_lr_pairs(somite_lut, pair_index_col='Somite_Stage', side_col='Side'):
    left = somite_lut.filter(
        pl.col(side_col).str.to_lowercase().str.starts_with('l')
    ).sort(pair_index_col)

    right = somite_lut.filter(
        pl.col(side_col).str.to_lowercase().str.starts_with('r')
    ).sort(pair_index_col)

    left_dict = _build_coord_dict(left, pair_index_col)
    right_dict = _build_coord_dict(right, pair_index_col)
    paired_numbers = sorted(set(left_dict.keys()) & set(right_dict.keys()))

    if len(paired_numbers) < 2:
        raise ValueError(
            f"Need at least 2 paired L/R somites, found {len(paired_numbers)}. "
            f"Left: {sorted(left_dict.keys())}, Right: {sorted(right_dict.keys())}"
        )

    midpoints = []
    lr_vectors = []
    for num in paired_numbers:
        L, R = left_dict[num], right_dict[num]
        midpoints.append((L + R) / 2)
        lr_vectors.append(R - L)

    return paired_numbers, np.array(midpoints), np.array(lr_vectors)


def _orthogonalize_lr(T_dense, lr_dense):
    dot = np.sum(lr_dense * T_dense, axis=1, keepdims=True)
    lr_perp = lr_dense - dot * T_dense
    norms = np.linalg.norm(lr_perp, axis=1, keepdims=True)

    degenerate = norms.ravel() < 1e-6
    if degenerate.any():
        logger.warning(f"{degenerate.sum()} points have LR nearly parallel to T")
        lr_perp[degenerate] = lr_dense[degenerate]
        norms[degenerate] = np.linalg.norm(lr_perp[degenerate], axis=1, keepdims=True)

    lr_perp /= norms

    dv = np.cross(T_dense, lr_perp)
    dv /= np.maximum(np.linalg.norm(dv, axis=1, keepdims=True), 1e-9)

    return lr_perp, dv, dot


def _find_closest_u(points, tck, kdt, u_dense, bracket_delta=0.05):
    u_params = np.zeros(len(points))
    for i, p in enumerate(points):
        _, idx = kdt.query(p)
        u_seed = u_dense[idx]
        bracket = (max(0, u_seed - bracket_delta), min(1, u_seed + bracket_delta))
        res = minimize_scalar(
            _distance_sq, bounds=bracket,
            args=(p, tck), method='bounded',
            options={'xatol': 1e-8}
        )
        u_params[i] = res.x if res.success else u_seed
    return u_params


# =============================================================================
# Frenet-Serret Frame
# =============================================================================

def calculate_frenet_frames(u_values, tck):
    u_values = np.asarray(u_values)
    is_scalar = u_values.ndim == 0
    if is_scalar:
        u_values = u_values[np.newaxis]

    s_prime = np.array(splev(u_values, tck, der=1)).T
    s_double_prime = np.array(splev(u_values, tck, der=2)).T

    if s_prime.ndim == 1:
        s_prime = s_prime[np.newaxis, :]
    if s_double_prime.ndim == 1:
        s_double_prime = s_double_prime[np.newaxis, :]

    norms = np.linalg.norm(s_prime, axis=1)
    T = np.zeros_like(s_prime)
    valid_T = norms > 1e-9
    T[valid_T] = s_prime[valid_T] / norms[valid_T, np.newaxis]

    B_raw = np.cross(T, s_double_prime, axis=1)
    norms_B = np.linalg.norm(B_raw, axis=1)
    B = np.zeros_like(B_raw)
    valid_B = norms_B > 1e-9
    B[valid_B] = B_raw[valid_B] / norms_B[valid_B, np.newaxis]

    N_raw = np.cross(B, T, axis=1)
    norms_N = np.linalg.norm(N_raw, axis=1)
    N = np.zeros_like(N_raw)
    valid_N = norms_N > 1e-9
    N[valid_N] = N_raw[valid_N] / norms_N[valid_N, np.newaxis]

    if is_scalar:
        return (T.squeeze(), N.squeeze(), B.squeeze(),
                valid_T.squeeze(), valid_N.squeeze(), valid_B.squeeze())
    return T, N, B, valid_T, valid_N, valid_B


def compute_frenet_frame(spline_points, smoothing=2.0, u_resolution=200):
    if spline_points.shape[1] != 3:
        raise ValueError(f"Expected (N, 3), got {spline_points.shape}")

    k = min(3, len(spline_points) - 1)
    tck, _ = splprep(spline_points.T, s=smoothing, k=k)

    u_dense = np.linspace(0, 1, u_resolution)
    curve_dense = np.array(splev(u_dense, tck)).T
    T, N, B, valid_T, valid_N, valid_B = calculate_frenet_frames(u_dense, tck)

    logger.info(f"Frenet frame: {len(spline_points)} control points, "
                f"{(~valid_N).sum()} undefined normals")

    return FrenetFrame(
        tck=tck, u_dense=u_dense, curve_dense=curve_dense,
        T=T, N=N, B=B,
        valid_T=valid_T, valid_N=valid_N, valid_B=valid_B,
        kdt=cKDTree(curve_dense)
    )


def compute_frenet_from_csv(spline_csv_path, smoothing=2.0, u_resolution=1000,
                            axis_cols=('axis-0', 'axis-1', 'axis-2')):
    df = pl.read_csv(spline_csv_path)
    pts = np.column_stack([df[c].to_numpy() for c in axis_cols])
    pts = to_s0(pts, S4_SCALE)
    return compute_frenet_frame(pts, smoothing=smoothing, u_resolution=u_resolution)


def frenet_frame_at_u(u, frame):
    idx = np.abs(frame.u_dense - u).argmin()
    T, N, B = frame.T[idx].ravel(), frame.N[idx].ravel(), frame.B[idx].ravel()
    valid = frame.valid_T[idx] and frame.valid_N[idx] and frame.valid_B[idx]
    return T, N, B, valid


def project_points_to_frenet(centroids, frame, return_frames=False,
                             seeds_k=5, bracket_delta=0.07):
    centroids = np.atleast_2d(centroids)
    n = len(centroids)

    u_params = np.zeros(n)
    tangent_coords = np.zeros(n)
    normal_coords = np.zeros(n)
    binormal_coords = np.zeros(n)
    if return_frames:
        local_frames = np.zeros((n, 3, 3))

    for i, p in enumerate(centroids):
        _, idxs = frame.kdt.query(p, k=seeds_k)
        idxs = np.atleast_1d(idxs)

        best_u, best_val = None, np.inf
        for idx in idxs:
            u_seed = float(frame.u_dense[int(idx)])
            a = max(0.0, u_seed - bracket_delta)
            b = min(1.0, u_seed + bracket_delta)
            if b - a < 1e-6:
                a, b = max(0.0, u_seed - 0.02), min(1.0, u_seed + 0.02)

            res = minimize_scalar(
                _distance_sq, bounds=(a, b),
                args=(p, frame.tck), method='bounded',
                options={'xatol': 1e-8, 'maxiter': 200}
            )
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_u = float(res.x)

        if best_u is None:
            best_u = float(frame.u_dense[int(idxs[0])])

        p_proj = np.array(splev(best_u, frame.tck))
        T, N_vec, B, valid = frenet_frame_at_u(best_u, frame)
        V = p - p_proj

        u_params[i] = best_u
        tangent_coords[i] = np.dot(V, T) if valid else 0.0
        normal_coords[i] = np.dot(V, N_vec) if valid else 0.0
        binormal_coords[i] = np.dot(V, B) if valid else 0.0
        if return_frames:
            local_frames[i] = np.column_stack([T, N_vec, B])

    result = {
        'u_param': u_params,
        'tangent_coord': tangent_coords,
        'normal_coord': normal_coords,
        'binormal_coord': binormal_coords
    }
    if return_frames:
        result['local_frames'] = local_frames
    return result


# =============================================================================
# Somite-based Anatomical Frame
# =============================================================================

def anatomical_frame_at_u(u, frame):
    idx = np.abs(frame.u_dense - u).argmin()
    return frame.T[idx].ravel(), frame.LR[idx].ravel(), frame.DV[idx].ravel()


def get_somite_centroids(somite_mask, somite_lut, scale=True):
    regions = measure.regionprops(somite_mask)
    labels, czs, cys, cxs = [], [], [], []
    for prop in regions:
        labels.append(prop.label)
        cz, cy, cx = prop.centroid
        if scale:
            czs.append(to_s0(cz, S3_SCALE))
            cys.append(to_s0(cy, S3_SCALE))
            cxs.append(to_s0(cx, S3_SCALE))
        else:
            czs.append(cz)
            cys.append(cy)
            cxs.append(cx)

    centroids_df = pl.DataFrame({
        'Somite_ID': labels,
        'Centroid_Z': czs, 
        'Centroid_Y': cys, 
        'Centroid_X': cxs
    })
    return somite_lut.join(centroids_df, on='Somite_ID', how='inner')


def compute_anatomical_frame(somite_lut, pair_index_col='Somite_Stage',
                             side_col='Side', smoothing=5.0,
                             u_resolution=1000):
    paired_numbers, midpoints, lr_vectors = _extract_lr_pairs(
        somite_lut, pair_index_col, side_col
    )
    logger.info(f"Found {len(paired_numbers)} paired somites: {paired_numbers}")

    k = min(3, len(midpoints) - 1)
    tck, u_somites = splprep(midpoints.T, s=smoothing, k=k)

    u_dense = np.linspace(0, 1, u_resolution)
    curve_dense = np.array(splev(u_dense, tck)).T

    tangents = np.array(splev(u_dense, tck, der=1)).T
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)

    lr_unit = lr_vectors / np.linalg.norm(lr_vectors, axis=1, keepdims=True)
    f_lr = interp1d(
        u_somites, lr_unit, axis=0, kind='linear',
        bounds_error=False, fill_value=(lr_unit[0], lr_unit[-1])
    )
    lr_dense = f_lr(u_dense)

    lr_perp, dv, _ = _orthogonalize_lr(tangents, lr_dense)

    return AnatomicalFrame(
        tck=tck, u_dense=u_dense, curve_dense=curve_dense,
        T=tangents, LR=lr_perp, DV=dv,
        kdt=cKDTree(curve_dense),
        paired_numbers=paired_numbers, u_somites=u_somites, midpoints=midpoints
    )


def project_points_to_anatomy(centroids, frame, return_frames=False):
    centroids = np.atleast_2d(centroids)
    n = len(centroids)

    pos_AP = np.zeros(n)
    pos_LR = np.zeros(n)
    pos_DV = np.zeros(n)
    if return_frames:
        local_frames = np.zeros((n, 3, 3))

    for i, p in enumerate(centroids):
        _, idx = frame.kdt.query(p)
        u_seed = frame.u_dense[idx]

        bracket = (max(0, u_seed - 0.05), min(1, u_seed + 0.05))
        res = minimize_scalar(
            _distance_sq, bounds=bracket,
            args=(p, frame.tck), method='bounded'
        )
        u_opt = res.x if res.success else u_seed

        p_proj = np.array(splev(u_opt, frame.tck))
        T, LR, DV = anatomical_frame_at_u(u_opt, frame)
        V = p - p_proj

        pos_AP[i] = u_opt
        pos_LR[i] = np.dot(V, LR)
        pos_DV[i] = np.dot(V, DV)
        if return_frames:
            local_frames[i] = np.column_stack([T, LR, DV])

    result = {'pos_AP': pos_AP, 'pos_LR': pos_LR, 'pos_DV': pos_DV}
    if return_frames:
        result['local_frames'] = local_frames
    return result


# =============================================================================
# Hybrid Frame (Recommended)
# =============================================================================

def compute_hybrid_frame(somite_lut, spline_points,
                         pair_index_col='Somite_Stage', side_col='Side',
                         spline_smoothing=2.0, u_resolution=1000):
    # Neural tube spline (spline_points must already be in s0)
    if spline_points.shape[1] != 3:
        raise ValueError(f"spline_points must be (N, 3), got {spline_points.shape}")
    if len(spline_points) < 2:
        raise ValueError("Need at least 2 spline points")

    k_nt = min(3, len(spline_points) - 1)
    tck_nt, _ = splprep(spline_points.T, s=spline_smoothing, k=k_nt)
    logger.info(f"Neural tube spline: {len(spline_points)} control points, k={k_nt}")

    # L/R somite pairs
    paired_numbers, midpoints, lr_vectors = _extract_lr_pairs(
        somite_lut, pair_index_col, side_col
    )
    logger.info(f"Found {len(paired_numbers)} paired somites: {paired_numbers}")

    # Dense evaluation along neural tube
    u_dense = np.linspace(0, 1, u_resolution)
    curve_dense = np.array(splev(u_dense, tck_nt)).T
    kdt_nt = cKDTree(curve_dense)

    # Map somite midpoints onto neural tube parameterization
    u_somites = _find_closest_u(midpoints, tck_nt, kdt_nt, u_dense)
    logger.info(f"Somite u-params on neural tube: [{u_somites.min():.3f}, {u_somites.max():.3f}]")

    # Frenet tangent along neural tube
    T_dense, _, _, valid_T, _, _ = calculate_frenet_frames(u_dense, tck_nt)
    if not valid_T.all():
        logger.warning(f"{(~valid_T).sum()} points have undefined tangent")

    # Interpolate LR vectors along neural tube
    lr_unit = lr_vectors / np.linalg.norm(lr_vectors, axis=1, keepdims=True)

    for i in range(1, len(lr_unit)):
        if np.dot(lr_unit[i], lr_unit[i - 1]) < 0:
            logger.warning(
                f"LR vector flips between stages {paired_numbers[i-1]} and {paired_numbers[i]}"
            )

    f_lr = interp1d(
        u_somites, lr_unit, axis=0, kind='linear',
        bounds_error=False, fill_value=(lr_unit[0], lr_unit[-1])
    )
    lr_dense = f_lr(u_dense)

    # Orthogonalize
    lr_perp, dv, dot = _orthogonalize_lr(T_dense, lr_dense)
    logger.info(f"Hybrid frame built. Max |LR·T|: {np.abs(dot).max():.4f}")

    return AnatomicalFrame(
        tck=tck_nt, u_dense=u_dense, curve_dense=curve_dense,
        T=T_dense, LR=lr_perp, DV=dv,
        kdt=kdt_nt,
        paired_numbers=paired_numbers, u_somites=u_somites, midpoints=midpoints
    )


# =============================================================================
# Orientation Features
# =============================================================================

def compute_orientation_features(major_axis_image, local_frame, eigenvalues=None):
    major_axis_image = major_axis_image / np.linalg.norm(major_axis_image)
    primary = local_frame.T @ major_axis_image

    if primary[0] < 0:
        primary = -primary

    align_AP, align_LR, align_DV = abs(primary[0]), abs(primary[1]), abs(primary[2])

    result = {
        'orient_AP': primary[0],
        'orient_LR': primary[1],
        'orient_DV': primary[2],
        'align_AP': align_AP,
        'align_LR': align_LR,
        'align_DV': align_DV,
        'theta_from_AP': np.arccos(np.clip(align_AP, 0, 1)),
        'phi_in_LR_DV': np.arctan2(abs(primary[2]), abs(primary[1]))
    }

    if eigenvalues is not None:
        eig_sorted = np.sort(eigenvalues)
        result['eccentricity'] = 1 - (eig_sorted[0] / eig_sorted[-1]) if eig_sorted[-1] > 0 else 0.0

    return result


def principal_axis_in_anatomy(mask, local_frame):
    vox = np.column_stack(np.where(mask))
    if len(vox) < 4:
        return {k: np.nan for k in [
            'orient_AP', 'orient_LR', 'orient_DV',
            'align_AP', 'align_LR', 'align_DV',
            'theta_from_AP', 'phi_in_LR_DV', 'eccentricity'
        ]}

    centered = vox - vox.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    major_axis = eigenvectors[:, np.argmax(eigenvalues)]

    return compute_orientation_features(major_axis, local_frame, eigenvalues)


# =============================================================================
# Visualization
# =============================================================================

def plot_anatomical_frame(frame, ax=None, n_frames=20, scale=50.0, title=None):
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

    curve = frame.curve_dense
    ax.plot(curve[:, 2], curve[:, 1], curve[:, 0],
            color='purple', linestyle='--', linewidth=3, label='Midline/Neural tube')
    ax.scatter(frame.midpoints[:, 2], frame.midpoints[:, 1], frame.midpoints[:, 0],
               c='gray', s=100, marker='o', label='L/R midpoints')

    indices = np.linspace(0, len(frame.u_dense) - 1, n_frames, dtype=int)
    colors = {'T': 'red', 'LR': 'green', 'DV': 'blue'}
    labels = {'T': 'AP (tangent)', 'LR': 'Left-Right', 'DV': 'Dorsal-Ventral'}

    for idx_plot, i in enumerate(indices):
        origin = curve[i]
        for vec, name in [(frame.T[i], 'T'), (frame.LR[i], 'LR'), (frame.DV[i], 'DV')]:
            ax.quiver(origin[2], origin[1], origin[0],
                      vec[2]*scale, vec[1]*scale, vec[0]*scale,
                      color=colors[name], arrow_length_ratio=0.1, linewidth=2,
                      label=labels[name] if idx_plot == 0 else "")

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    ax.set_title(title or 'Anatomical Coordinate System')
    return fig


def plot_frame_3d(frame, somite_centroids=None, n_frames=20, scale=1.0, title=None):
    import plotly.graph_objects as go

    fig = go.Figure()
    curve = frame.curve_dense

    fig.add_trace(go.Scatter3d(
        x=curve[:, 2], y=curve[:, 1], z=curve[:, 0],
        mode="lines", line=dict(color="purple", width=4, dash="dash"),
        name="Midline / Neural tube",
    ))
    fig.add_trace(go.Scatter3d(
        x=frame.midpoints[:, 2], y=frame.midpoints[:, 1], z=frame.midpoints[:, 0],
        mode="markers", marker=dict(size=5, color="gray"),
        name="L/R midpoints",
    ))

    if somite_centroids is not None:
        sc = np.asarray(somite_centroids)
        fig.add_trace(go.Scatter3d(
            x=sc[:, 2], y=sc[:, 1], z=sc[:, 0],
            mode="markers", marker=dict(size=4, color="orange"),
            name="Somite centroids",
        ))

    indices = np.linspace(0, len(frame.u_dense) - 1, n_frames, dtype=int)
    axis_meta = {
        "T":  {"color": "red",   "label": "AP (tangent)"},
        "LR": {"color": "green", "label": "Left-Right"},
        "DV": {"color": "blue",  "label": "Dorsal-Ventral"},
    }

    for name, meta in axis_meta.items():
        xs, ys, zs = [], [], []
        ox, oy, oz, ux, uy, uz = [], [], [], [], [], []
        for i in indices:
            origin = curve[i]
            vec = getattr(frame, name)[i]
            tip = origin + vec * scale
            xs.extend([origin[2], tip[2], None])
            ys.extend([origin[1], tip[1], None])
            zs.extend([origin[0], tip[0], None])
            ox.append(tip[2]); oy.append(tip[1]); oz.append(tip[0])
            ux.append(vec[2]); uy.append(vec[1]); uz.append(vec[0])

        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color=meta["color"], width=3), name=meta["label"],
        ))
        fig.add_trace(go.Cone(
            x=ox, y=oy, z=oz, u=ux, v=uy, w=uz,
            sizemode="absolute", sizeref=scale*0.25, anchor="tail",
            showscale=False, colorscale=[[0, meta["color"]], [1, meta["color"]]],
            name=meta["label"], showlegend=False,
        ))

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        legend=dict(itemsizing="constant"), width=900, height=750,
    )
    return fig


def plot_frenet_frame(frame, somite_mask, ax=None, n_frames=200, scale_vecs=50.0):
    import matplotlib.pyplot as plt
    from scipy import ndimage as ndi

    if ax is None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

    mask_labels = np.unique(somite_mask)
    mask_labels = mask_labels[mask_labels != 0]

    if mask_labels.size > 0:
        com_zyx = np.array(ndi.center_of_mass(
            np.ones_like(somite_mask, dtype=np.float32),
            labels=somite_mask, index=mask_labels.tolist(),
        ))
        ax.scatter(
            to_s0(com_zyx[:, 2] + 0.5, S3_SCALE),
            to_s0(com_zyx[:, 1] + 0.5, S3_SCALE),
            to_s0(com_zyx[:, 0] + 0.5, S3_SCALE),
            s=35, c="yellow", edgecolor="black", linewidth=0.5,
            label="Somite centroids",
        )

    curve = frame.curve_dense
    ax.plot(curve[:, 2], curve[:, 1], curve[:, 0],
            color='purple', linestyle='-', linewidth=2, label='Spline curve')

    indices = np.linspace(0, len(frame.u_dense) - 1, n_frames, dtype=int)
    colors = {'T': 'red', 'N': 'green', 'B': 'blue'}
    labels = {'T': 'Tangent', 'N': 'Normal', 'B': 'Binormal'}

    for idx_plot, i in enumerate(indices):
        if not (frame.valid_T[i] and frame.valid_N[i] and frame.valid_B[i]):
            continue
        origin = curve[i]
        for vec, name in [(frame.T[i], 'T'), (frame.N[i], 'N'), (frame.B[i], 'B')]:
            ax.quiver(origin[2], origin[1], origin[0],
                      vec[2]*scale_vecs, vec[1]*scale_vecs, vec[0]*scale_vecs,
                      color=colors[name], arrow_length_ratio=0.1, linewidth=2,
                      label=labels[name] if idx_plot == 0 else "")

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('Frenet-Serret Frame (Geometric)')
    return fig
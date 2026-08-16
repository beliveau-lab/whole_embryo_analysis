"""
Single source of truth for figure saving, color helpers, and camera utilities.
Use this module instead of duplicating save_vectorized or hex_to_rgb elsewhere.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


def save_vectorized(
    dir_path: Union[str, Path, None],
    fig: "plt.Figure",
    plot_name: str,
    transparent: bool = True,
) -> None:
    """Save figure as SVG; create directory if needed. If dir_path is None, show instead."""
    text_color = "white" if transparent else "black"
    for ax in fig.get_axes():
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.title.set_color(text_color)
        ax.tick_params(colors=text_color, which="both")
        for spine in ax.spines.values():
            spine.set_edgecolor(text_color)
        if ax.get_legend():
            ax.get_legend().get_frame().set_edgecolor(text_color)
            for text in ax.get_legend().get_texts():
                text.set_color(text_color)
    for text in fig.texts:
        text.set_color(text_color)

    output_path = Path(dir_path) if dir_path else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        out_file = output_path / f"{plot_name}.svg"
        fig.savefig(
            out_file,
            format="svg",
            bbox_inches="tight",
            transparent=transparent,
        )
        logger.info("Saved plot: %s", out_file)
    else:
        plt.show()


def napari_camera_to_plotly(
    angles: tuple[float, float, float],
    center: tuple[float, float, float],
    zoom: float,
    s_scale: float = 4.0,
    x_vals: np.ndarray | None = None,
    y_vals: np.ndarray | None = None,
    z_vals: np.ndarray | None = None,
) -> dict:
    """Convert Napari-style camera parameters to a Plotly ``scene_camera`` dict.

    Parameters
    ----------
    angles : (elevation, azimuth, roll)
        Napari camera angles in degrees.
    center : (z, y, x)
        Napari camera center in **s2 ZYX pixels**.
    zoom : float
        Napari zoom level.
    s_scale : float
        Downsampling factor for the center coordinates (e.g. 4 for s2).
    x_vals, y_vals, z_vals : array-like
        Data coordinates used to compute data range and centroid for
        normalization.  All three are required.
    """
    ang = np.deg2rad(angles)

    view_direction = np.array([
        np.sin(ang[2]) * np.cos(ang[1]),
        np.cos(ang[2]) * np.cos(ang[1]),
        -np.sin(ang[1]),
    ])[::-1]

    R = Rotation.from_euler('yzx', angles, degrees=True).as_matrix()
    up_direction = (R[:, 2][::-1])[::-1]

    s0_center = np.array(center)[::-1] * s_scale

    data_range = max(np.ptp(x_vals), np.ptp(y_vals), np.ptp(z_vals))
    distance = data_range / zoom
    eye_pos = s0_center - view_direction * distance

    data_mid = np.array([x_vals.mean(), y_vals.mean(), z_vals.mean()])
    eye_norm = (eye_pos - data_mid) / data_range
    center_norm = (s0_center - data_mid) / data_range

    return dict(
        eye=dict(x=float(eye_norm[0]), y=float(eye_norm[1]), z=float(eye_norm[2])),
        up=dict(x=float(up_direction[0]), y=float(up_direction[1]), z=float(up_direction[2])),
        center=dict(x=float(center_norm[0]), y=float(center_norm[1]), z=float(center_norm[2])),
        projection=dict(type="orthographic"),
    )


def hex_to_rgb(hex_color: str, normalize: bool = False) -> Union[tuple[int, int, int], np.ndarray]:
    """Convert hex string (e.g. '#rrggbb' or 'rrggbb') to RGB.

    Parameters
    ----------
    hex_color : str
    normalize : bool
        If False (default), return tuple (r, g, b) in 0–255.
        If True, return array in [0, 1] (float).

    Returns
    -------
    tuple or np.ndarray
    """
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    if normalize:
        return np.array([r, g, b], dtype=np.float64) / 255.0
    return (r, g, b)


def all_hex_to_rgb(hex_colors: list[str], normalize: bool = False) -> list:
    """Convert list of hex strings to list of RGB (tuple or array depending on normalize)."""
    return [hex_to_rgb(h, normalize=normalize) for h in hex_colors]

"""
Single source of truth for figure saving and color helpers.
Use this module instead of duplicating save_vectorized or hex_to_rgb elsewhere.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def save_vectorized(
    dir_path: Union[str, Path, None],
    fig: "plt.Figure",
    plot_name: str,
    *,
    transparent: bool = False,
) -> None:
    """Save figure as SVG; create directory if needed. If dir_path is None, show instead."""
    output_path = Path(dir_path) if dir_path else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        out_file = output_path / f"{plot_name}.svg"
        fig.savefig(
            out_file,
            format="svg",
            bbox_inches="tight",
            transparent=transparent,
            facecolor=None if transparent else "white",
            edgecolor="white" if transparent else "black",
        )
        logger.info("Saved plot: %s", out_file)
    else:
        plt.show()


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

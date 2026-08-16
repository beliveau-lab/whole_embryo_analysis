"""
Single source of truth for array/bbox I/O (N5/Zarr loading, chunk indices, crops).
Use this module instead of duplicating load_n5_zarr_array, add_chunk_indices,
get_bounds, load_data, or crop_max_proj elsewhere.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import zarr
import dask.array as da

logger = logging.getLogger(__name__)


def load_n5_zarr_array(
    path: str,
    n5_subpath: Optional[str] = None,
    chunks=None,
):
    """Load N5 or Zarr array as a Dask array.

    For .n5 uses zarr.N5Store; for .zarr uses zarr.open. Optional subpath
    (e.g. 'ch0/s0') for both formats.

    Returns
    -------
    dask.array.Array
    """
    logger.info("Attempting to load from: %s%s", path, f" with subpath: {n5_subpath}" if n5_subpath else "")
    if path.endswith(".n5"):
        store = zarr.N5Store(path)
        if not n5_subpath:
            raise ValueError(f"N5 path specified ({path}), but n5_subpath is required.")
        arr_handle = zarr.open_array(store=store, path=n5_subpath, mode="r")
        logger.info("Loaded N5 array: Shape=%s, Chunks=%s", arr_handle.shape, arr_handle.chunks)
        return da.from_zarr(arr_handle, chunks=chunks)
    if path.endswith(".zarr"):
        if n5_subpath:
            store = zarr.open(path, mode="r")
            arr_handle = store[n5_subpath]
            logger.info("Loaded Zarr array from subpath %r: Shape=%s", n5_subpath, arr_handle.shape)
            return da.from_zarr(arr_handle, chunks=chunks)
        arr_handle = zarr.open(path, mode="r")
        logger.info("Loaded Zarr array: Shape=%s", arr_handle.shape)
        return da.from_zarr(path, chunks=chunks)
    raise ValueError(f"Unsupported array format (expected .n5 or .zarr): {path}")


# =============================================================================
# Feature-table I/O (CSV or Parquet, transparent dispatch)
# =============================================================================

_PARQUET_SUFFIXES = (".parquet", ".pq")


def _is_parquet(path: str) -> bool:
    return str(path).lower().endswith(_PARQUET_SUFFIXES)


def scan_features(path: str) -> pl.LazyFrame:
    """Return a polars LazyFrame for either a .csv or .parquet feature table.

    Prefer this over calling ``pl.scan_csv`` directly so the pipeline can
    transparently consume a Parquet-converted feature table (which is far
    faster to load for 3TB-scale samples).
    """
    path = str(path)
    if _is_parquet(path):
        return pl.scan_parquet(path)
    return pl.scan_csv(path)


def read_features(path: str) -> pl.DataFrame:
    """Eagerly read a feature table in either .csv or .parquet format."""
    path = str(path)
    if _is_parquet(path):
        return pl.read_parquet(path)
    return pl.read_csv(path)


def add_chunk_indices(df_bboxes, chunk_sizes: tuple):
    """Add chunk index columns (cz, cy, cx) using bbox centroid and given chunk sizes.

    Accepts either a pandas or polars DataFrame and returns the same type.
    """
    cz_size = int(chunk_sizes[0][0])
    cy_size = int(chunk_sizes[1][0])
    cx_size = int(chunk_sizes[2][0])

    if isinstance(df_bboxes, pl.DataFrame):
        return df_bboxes.with_columns(
            (((pl.col("z0") + pl.col("z1")) // 2) // cz_size).cast(pl.Int64).alias("cz"),
            (((pl.col("y0") + pl.col("y1")) // 2) // cy_size).cast(pl.Int64).alias("cy"),
            (((pl.col("x0") + pl.col("x1")) // 2) // cx_size).cast(pl.Int64).alias("cx"),
        )

    df_bboxes = df_bboxes.copy()
    df_bboxes["cz"] = (((df_bboxes["z0"] + df_bboxes["z1"]) // 2) // cz_size).astype(np.int64)
    df_bboxes["cy"] = (((df_bboxes["y0"] + df_bboxes["y1"]) // 2) // cy_size).astype(np.int64)
    df_bboxes["cx"] = (((df_bboxes["x0"] + df_bboxes["x1"]) // 2) // cx_size).astype(np.int64)
    return df_bboxes


def get_bounds(group_df) -> tuple[int, int, int, int, int, int]:
    """Return (zstart, zend, ystart, yend, xstart, xend) from bbox columns.

    Accepts either a pandas or polars DataFrame.
    """
    zstart = int(group_df["z0"].min())
    zend = int(group_df["z1"].max())
    ystart = int(group_df["y0"].min())
    yend = int(group_df["y1"].max())
    xstart = int(group_df["x0"].min())
    xend = int(group_df["x1"].max())
    return zstart, zend, ystart, yend, xstart, xend


def load_data(
    mask_path: str,
    image_zarr_root: str,
    channels: list[int],
    chunks=None,
    n5_subpath: Optional[str] = None,
):
    """Load mask as Dask array and image channels as dict of zarr arrays."""
    logger.info("Loading mask from: %s", mask_path)
    mask_array = load_n5_zarr_array(mask_path, n5_subpath=n5_subpath, chunks=chunks)
    logger.info(
        "Loaded Mask (ZYX assumed): Shape=%s, Chunks=%s, Dtype=%s",
        mask_array.shape, mask_array.chunks, mask_array.dtype,
    )
    image_arrays = {}
    logger.info("Loading image channels %s from: %s", channels, image_zarr_root)
    for ch in channels:
        ch_path = os.path.join(image_zarr_root, f"ch{ch}", "s0")
        logger.info("Loading channel %s from: %s", ch, ch_path)
        try:
            ch_arr = zarr.open(ch_path, mode="r")
            image_arrays[ch] = ch_arr
            logger.info("Loaded channel %s: Shape=%s, Dtype=%s", ch, ch_arr.shape, ch_arr.dtype)
        except Exception as e:
            logger.error("Failed to load channel %s from %s: %s", ch, ch_path, e)
            raise
    return mask_array, image_arrays


def crop_max_proj(
    image_path: str,
    mask_path: str,
    centroid_z: float,
    centroid_y: float,
    centroid_x: float,
    image_subpath: str = "ch2/s0",
    mask_subpath: Optional[str] = None,
    crop_size: int = 128,
) -> np.ndarray:
    """Crop a 3D volume around (centroid_z, centroid_y, centroid_x), mask by object at that voxel, return crop.

    Image is loaded from image_path (N5 or Zarr) with image_subpath (e.g. 'ch2/s0').
    Mask is loaded from mask_path; use mask_subpath if the array is inside a group.
    Returns a 3D numpy array (masked crop) suitable for max projection.
    """
    if image_path.endswith(".n5"):
        store = zarr.N5Store(image_path)
        arr = zarr.open_array(store=store, path=image_subpath, mode="r")
    else:
        store = zarr.open(image_path, mode="r")
        arr = store[image_subpath] if image_subpath else store

    z_dim, y_dim, x_dim = arr.shape
    x, y, z = int(centroid_x), int(centroid_y), int(centroid_z)

    if mask_path.endswith(".n5"):
        mask_store = zarr.N5Store(mask_path)
        mask = zarr.open_array(store=mask_store, path=mask_subpath or "s0", mode="r")
    else:
        mask_store = zarr.open(mask_path, mode="r")
        mask = mask_store[mask_subpath] if mask_subpath else mask_store
        if not hasattr(mask, "shape"):
            keys = list(mask_store.keys()) if hasattr(mask_store, "keys") else []
            mask = mask_store[keys[0]] if keys else mask_store

    half = crop_size // 2
    x1, x2 = max(0, x - half), min(x_dim, x + half)
    y1, y2 = max(0, y - half), min(y_dim, y + half)
    z1, z2 = max(0, z - half), min(z_dim, z + half)

    obj_id = mask[z, y, x]
    binary_mask = mask[z1:z2, y1:y2, x1:x2] == obj_id
    img_crop = np.asarray(arr[z1:z2, y1:y2, x1:x2])
    img_crop = np.where(binary_mask, img_crop, 0)
    return img_crop

import numpy as np

S1_SCALE = 2
S2_SCALE = 4
S3_SCALE = 8    # somite masks drawn at s3 (8x downsample)
S4_SCALE = 16   # spline points drawn at s4 (16x downsample)

ROUND_OPS = {
    "floor": np.floor, # best for working with centroids/indexing masks
    "round": np.round, # round/ceil are better for spatial extents or bounding boxes
    "ceil": np.ceil,
}

from scipy.ndimage import zoom

def _scale_coords(coords, scale, operation, as_index=False, rounding="floor"):
    """Shared coordinate scaling for scalar or array-like values."""
    if coords is None:
        return None
    if as_index and rounding not in ROUND_OPS:
        raise ValueError("rounding must be one of: floor, round, ceil")

    if np.isscalar(coords):
        try:
            scaled = operation(float(coords), scale)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(scaled):
            return None
        if not as_index:
            return scaled
        return int(ROUND_OPS[rounding](scaled))

    scaled = operation(np.asarray(coords, dtype=float), scale)
    if not as_index:
        return scaled
    if not np.isfinite(scaled).all():
        raise ValueError("Non-finite values found; cannot convert to integer indices.")
    return ROUND_OPS[rounding](scaled).astype(np.int64)


def to_s0(coords, scale, as_index=False, rounding="floor"):
    """
    Convert coordinates from a downsampled space to s0 (original resolution).

    Set ``as_index=True`` to return integer voxel indices.
    """
    return _scale_coords(
        coords,
        scale,
        operation=lambda values, factor: values * factor,
        as_index=as_index,
        rounding=rounding,
    )

def from_s0(coords, scale, as_index=False, rounding="floor"):
    """
    Convert s0 coordinates to a downsampled space.

    Parameters
    ----------
    coords : scalar or array-like
        Coordinate(s) in s0 space.
    scale : int or float
        Downsampling factor (e.g., 8 for s3).
    as_index : bool, default False
        If True, return integer voxel indices for array indexing.
    rounding : {"floor", "round", "ceil"}, default "floor"
        Rounding strategy used when ``as_index=True``.
    """
    return _scale_coords(
        coords,
        scale,
        operation=lambda values, factor: values / factor,
        as_index=as_index,
        rounding=rounding,
    )

def scale_mask(mask, scale):
    """
    Scale a mask to a different resolution.
    """
    return zoom(mask, scale, order=0)
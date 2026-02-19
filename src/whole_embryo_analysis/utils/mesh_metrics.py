"""
Mesh comparison utilities (single source of truth).
Credit: https://github.com/AllenCell/cvapipe_analysis/tree/main
"""
from __future__ import annotations

import numpy as np
import scipy.spatial as spatial
from vtk.util import numpy_support as vtknp


def get_meshes_distance(mesh1, mesh2):
    """Compute bidirectional closest-point distances between two VTK meshes.

    Returns
    -------
    d12 : np.ndarray
        For each point in mesh2, distance to nearest point in mesh1.
    d21 : np.ndarray
        For each point in mesh1, distance to nearest point in mesh2.
    """
    coords1 = vtknp.vtk_to_numpy(mesh1.GetPoints().GetData())
    coords2 = vtknp.vtk_to_numpy(mesh2.GetPoints().GetData())
    dist = spatial.distance.cdist(coords1, coords2)
    d12 = dist.min(axis=0)
    d21 = dist.min(axis=1)
    return d12, d21

import os
import random 
import numpy as np
import pandas as pd
import zarr
import dask.array as da
import pyvista as pv
from skimage import measure
from sklearn.decomposition import PCA
from aicsshparam.shparam import get_shcoeffs
from aicsshparam.shtools import (
    get_mesh_from_image, get_reconstruction_from_coeffs, convert_coeffs_dict_to_matrix, voxelize_meshes
)
import plotly.graph_objects as go
import polars as pl
from colorcet import glasbey_dark
import logging

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from align_3d import (
    zyx_to_xyz, align_object_standard
    )

logger = logging.getLogger(__name__)

COLOR_DICT = {1: glasbey_dark[0], 2: glasbey_dark[2], 3: glasbey_dark[10]}
NUCLEAR_COLOR = glasbey_dark[43]

def load_n5_zarr_array(path, n5_subpath=None, chunks=None):
    logger.info(f"Attempting to load from: {path}" + (f" with subpath: {n5_subpath}" if n5_subpath else ""))
    if path.endswith('.n5'):
        store = zarr.open(path, mode='r')
        if not n5_subpath:
             raise ValueError(f"N5 path specified ({path}), but n5_subpath is required.")
        arr_handle = zarr.open_array(store=store, path=n5_subpath, mode='r')
        logger.info(f"Loaded N5 array: Shape={arr_handle.shape}, Chunks={arr_handle.chunks}")
        return da.from_zarr(arr_handle, chunks=chunks)
    elif path.endswith('.zarr'):
        if n5_subpath:
            # open zarr store and access the specific subpath using dictionary access #
            store = zarr.open(path, mode='r')
            arr_handle = store[n5_subpath]
            logger.info(f"Loaded Zarr array from subpath '{n5_subpath}': Shape={arr_handle.shape}")
            return da.from_zarr(arr_handle, chunks=chunks)
        else:
            arr_handle = zarr.open(path, mode='r')
            logger.info(f"Loaded Zarr array: Shape={arr_handle.shape}")
            return da.from_zarr(path, chunks=chunks)
    else:
        raise ValueError(f"Unsupported array format (expected .n5 or .zarr): {path}")

def load_data(mask_path, image_zarr_root, channels, chunks=None, n5_subpath=None):
    logger.info(f"Loading mask from: {mask_path}")
    mask_array = load_n5_zarr_array(mask_path, n5_subpath=n5_subpath, chunks=chunks)
    logger.info(f"Loaded Mask (ZYX assumed): Shape={mask_array.shape}, Chunks={mask_array.chunks}, Dtype={mask_array.dtype}")
    image_arrays = {}
    logger.info(f"Loading image channels {channels} from: {image_zarr_root}")
    for ch in channels:
        ch_path = os.path.join(image_zarr_root, f"ch{ch}", "s0")
        logger.info(f"Loading channel {ch} from: {ch_path}")
        try:
            ch_arr = zarr.open(ch_path, mode='r')
            image_arrays[ch] = ch_arr
            logger.info(f"Loaded channel {ch}: Shape={ch_arr.shape}, Dtype={ch_arr.dtype}")
        except Exception as e:
            logger.error(f"Failed to load channel {ch} from {ch_path}: {e}")
            raise
    
    return mask_array, image_arrays

def add_chunk_indices(df_bboxes: pd.DataFrame, chunk_sizes: tuple) -> pd.DataFrame:
    """Add chunk index columns (cz, cy, cx) using bbox centroid and given chunk sizes."""
    cz_size, cy_size, cx_size = (int(chunk_sizes[0][0]), int(chunk_sizes[1][0]), int(chunk_sizes[2][0]))
    cz = ((df_bboxes["z0"] + df_bboxes["z1"]) // 2) // cz_size
    cy = ((df_bboxes["y0"] + df_bboxes["y1"]) // 2) // cy_size
    cx = ((df_bboxes["x0"] + df_bboxes["x1"]) // 2) // cx_size
    df_bboxes = df_bboxes.copy()
    df_bboxes["cz"] = cz.astype(np.int64)
    df_bboxes["cy"] = cy.astype(np.int64)
    df_bboxes["cx"] = cx.astype(np.int64)
    return df_bboxes
    
def get_bounds(group_df):
    zstart = int(group_df["z0"].min()); zend = int(group_df["z1"].max())
    ystart = int(group_df["y0"].min()); yend = int(group_df["y1"].max())
    xstart = int(group_df["x0"].min()); xend = int(group_df["x1"].max())
    return zstart, zend, ystart, yend, xstart, xend

def get_principal_axes(img):
    coords = zyx_to_xyz(img)
    centroid = coords.mean(axis=0)
    X = coords - centroid
    pca = PCA(n_components=3).fit(X)
    E = pca.components_
    eigenvalues = pca.explained_variance_
    projections = np.dot(coords, E.T)
    lengths = np.ptp(projections, axis=0)
    return centroid, E, eigenvalues, lengths

def create_axes(centroid, E, eigenvalues, lengths, scale=20):
    # colors = ['red', 'green', 'blue']
    axes_traces = []
    names = ['Major Axis', 'Intermediate Axis', 'Minor Axis']
    for i in range(3):
        c = COLOR_DICT[i+1]
        start = centroid - lengths[i] / 2 * E[i]
        end = centroid + lengths[i] / 2 * E[i]
        positive_trace = go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode="lines+markers",
            line=dict(color=c, width=6),
            marker=dict(size=[4, 8], color=c),
            showlegend=False#,
            # name=names[i]
        )
        axes_traces.append(positive_trace)
        
    return axes_traces

def plot_object(obj_tuple, title, dir, eye_val=.5, show_title=False, save_image=False):
    mesh, img = obj_tuple
    padded_img = np.pad(img, pad_width=1, mode='constant', constant_values=0)
    vertices, faces, _, _ = measure.marching_cubes(padded_img)
    centroid, E, eigenvalues, lengths = get_principal_axes(img)
    x_coords, y_coords, z_coords = (
        vertices[:, 2], vertices[:, 1], vertices[:, 0]
    )
    v0, v1, v2 = (
        faces[:, 2], faces[:, 1], faces[:, 0]
    )
    mesh_plot = (
        go.Mesh3d(
            x=x_coords, y=y_coords, z=z_coords,
            i=v0, j=v1, k=v2,
            color="#1f77b4",
            opacity=0.5
        )
    )
    axes_traces = create_axes(centroid, E, eigenvalues, lengths)
    fig = go.Figure(data=[mesh_plot] + axes_traces)
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False),
        ),
        scene_camera=dict(
            eye=dict(x=eye_val, y=eye_val, z=eye_val),  # Example values

        ),
        title=title.upper() if show_title else None,
    )
    fig.update_scenes(camera_projection_type='orthographic')
    if save_image:
        fig.write_image(
            f"{dir}/{title.strip().replace(' ', '_')}.svg", 
            format='svg',
            engine='kaleido')
        logger.info(f"Saved image to: {dir}/{title.strip().replace(' ', '_')}.svg")
    else:
        fig.show()
    return mesh_plot, axes_traces

def get_coeffs(obj_id, features_path): 
    query = pl.scan_csv(features_path)
    schema = query.collect_schema()
    shcoeff_cols = [c for c in schema.names() if 'shcoeffs_' in c]
    query = (
        query
        .filter(pl.col("Nuclear_ID") == obj_id)
        .select(shcoeff_cols)
        .collect()
    )
    df = query.to_pandas()
    row = df.iloc[0]
    mat = np.array(convert_coeffs_dict_to_matrix(row.to_dict(), lmax=16), dtype=np.float32)
    return mat

def reconstruct(mat, lrec=0):
    # Ensure mat is numpy array
    if isinstance(mat, list):
        mat = np.array(mat, dtype=np.float32)
    # Get the reconstructed mesh
    mesh, grid = get_reconstruction_from_coeffs(mat, lrec=lrec)
    # Voxelize the mesh to create an image
    img, origin = voxelize_meshes([mesh])
    img = img.astype(np.uint8)
    return mesh, img

def plot_reconstruction(obj_id, title, dir, coeffs, show_title=False, lmax=16):
    mat = get_coeffs(obj_id, features_path)  # Fixed function name
    mesh_rec, img_rec = reconstruct(mat, lrec=lmax)
    rec_tuple = (mesh_rec, img_rec)
    plot_object(rec_tuple, title, dir, show_title=show_title)    
    return mesh_rec
    
def find_specific_object(obj_id, df_bboxes, id_name='Nuclear_ID'):
    df_bboxes = df_bboxes[df_bboxes[id_name] == obj_id]
    return df_bboxes
    
def get_object(obj_id, mask_roi):
    obj_image = np.where(mask_roi == obj_id, 1, 0)
    obj_image = obj_image.compute()
    mesh, _, _ = get_mesh_from_image(obj_image)
    return mesh, obj_image

def crop_max_proj(zarr_path, mask_path, centroid_z, centroid_y, centroid_x, crop_size=128):
    # Open the zarr file
    mask_da, image_arrays = load_data(mask_path, zarr_path, [2])
    arr = image_arrays[2]

    # Get the shape (z, y, x)
    z_dim, y_dim, x_dim = arr.shape

    # Crop in x and y around centroid_x, centroid_y
    # For this dataset, assume centroid_x is x, centroid_y is y
    # Use all z for max projection
    x = int(centroid_x)
    y = int(centroid_y)
    z = int(centroid_z)
    # If centroid_y is not available, use centroid_x for both (will be square)

    # use mask to crop
    mask = zarr.open_array(mask_path, mode='r')
    

    half = crop_size // 2
    x1, x2 = max(0, x - half), min(x_dim, x + half)
    y1, y2 = max(0, y - half), min(y_dim, y + half) 
    z1, z2 = max(0, z - half), min(z_dim, z + half)

    obj_id = mask[z, y, x]

    binary_mask = mask[z1:z2, y1:y2, x1:x2] == obj_id
    img_crop = arr[z1:z2, y1:y2, x1:x2]
    img_crop = np.where(binary_mask, img_crop, 0) # crop the array to the mask
    # Max projection over z
    return img_crop


    
# def visualize(mask, max_points=10000):
#     coords = align.get_zyx_coords(mask)
#     if len(coords) > max_points:
#         idx = np.random.choice(len(coords), max_points, replace=False)
#         coords = coords[idx]
#     fig = plt.figure(figsize=(15, 6))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(coords[:, 2], coords[:, 1], coords[:, 0], alpha=0.7, s=1)
#     ax.set_xlabel('Z')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('X')
#     ax.set_title('Rotated Object')
#     plt.show()


# def visualize_comparison(original, aligned, max_points=10000):
#     """
#     Create a 3D visualization comparing original and aligned objects.
    
#     Args:
#         original: Original boolean 3D array
#         aligned: Aligned boolean 3D array  
#         max_points: Maximum points to plot (for performance)
#     """
#     # Get coordinates for both objects
#     orig_coords = align.get_zyx_coords(original)
#     aligned_coords = align.get_zyx_coords(aligned)
#     # Subsample if too many points
#     if len(orig_coords) > max_points:
#         idx = np.random.choice(len(orig_coords), max_points, replace=False)
#         orig_coords = orig_coords[idx]
#     if len(aligned_coords) > max_points:
#         idx = np.random.choice(len(aligned_coords), max_points, replace=False)
#         aligned_coords = aligned_coords[idx]
#     # Create side-by-side 3D plots
#     fig = plt.figure(figsize=(15, 6))
#     # Original object
#     ax1 = fig.add_subplot(121, projection='3d')
#     ax1.scatter(orig_coords[:, 2], 
#                 orig_coords[:, 1], 
#                 orig_coords[:, 0], 
#                 c='red', 
#                 alpha=0.6, 
#                 s=1)
#     ax1.set_title('Original Object')
#     ax1.set_xlabel('Z')
#     ax1.set_ylabel('Y')
#     ax1.set_zlabel('X')
#     # Aligned object
#     ax2 = fig.add_subplot(122, projection='3d')
#     ax2.scatter(aligned_coords[:, 2], 
#                 aligned_coords[:, 1], 
#                 aligned_coords[:, 0], 
#                 c='blue', 
#                 alpha=0.6, s=1)
#     ax2.set_title('Aligned Object')
#     ax2.set_xlabel('Z')
#     ax2.set_ylabel('Y')
#     ax2.set_zlabel('X')
    
#     plt.tight_layout()
#     plt.show()


# def plot_mesh(mask, max_points=10000):
#     """Simple three-view visualization of 3D object with principal axes"""
#     # Get coordinates and subsample if needed
#     coords = align.get_zyx_coords(mask)
#     if len(coords) > max_points:
#         idx = np.random.choice(len(coords), max_points, replace=False)
#         coords = coords[idx]
#     # Get principal axes
#     major, intermediate, minor, lengths = align.get_principal_axes(coords, return_lengths=True)
#     centroid = coords.mean(axis=0)
#     # Setup three views
#     fig = plt.figure(figsize=(18, 6))
#     views = [
#         {'elev': 0, 'azim': 90, 'title': 'Side View'},
#         {'elev': 45, 'azim': 45, 'title': 'Isometric View'},
#         {'elev': 90, 'azim': 0, 'title': 'Top View'}
#     ]
#     for idx, view in enumerate(views):
#         ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
#         ax.scatter(coords[:, 2], 
#                    coords[:, 1], 
#                    coords[:, 0], 
#                    alpha=0.7, 
#                    s=1)
#         # plot principal axes
#         for vec, color, label, length in zip([major, intermediate, minor], 
#                                            ['r', 'g', 'b'], 
#                                            ['Major', 'Intermediate', 'Minor'],
#                                            lengths):
#             pt1 = centroid - length/2 * vec
#             pt2 = centroid + length/2 * vec
#             # Convert ZYX to XYZ for plotting
#             ax.plot([pt1[2], pt2[2]], [pt1[1], pt2[1]], [pt1[0], pt2[0]], 
#                    color=color, linewidth=3, label=label, alpha=0.5)
        
#         ax.view_init(elev=view['elev'], azim=view['azim'])
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_zlabel('Z')
#         ax.set_title(view['title'])
#         if idx == 0:
#             ax.legend()
    
#     plt.tight_layout()
#     plt.show()

# def plot_aligned_object(aligned_volume):
#     """
#     Plots a 3D mesh of an object after it has been aligned, with its
#     principal axes aligned with the coordinate system axes.

#     Args:
#         aligned_volume (np.ndarray): The 3D numpy array of the aligned object.
#     """
#     print("Plotting aligned object...")
#     xyz_coords = align.get_zyx_coords(aligned_volume)
#     major, minor, intermediate = align.get_principal_axes(xyz_coords)
#     # For an aligned object, principal axes correspond to the standard basis vectors.
#     print(f"\tPlotting with standard axes:")
#     print(f"\tMajor Axis: {major}")
#     print(f"\tIntermediate Axis: {intermediate}")
#     print(f"\tMinor Axis: {minor}")

#     plot_mesh(aligned_volume)
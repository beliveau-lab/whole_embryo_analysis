from __future__ import annotations

import time
import logging
from pathlib import Path

import numpy as np
import polars as pl
from scipy.ndimage import distance_transform_edt
from whole_embryo_analysis.somite_analysis.bio_frame import (
    AnatomicalFrame,
    FrenetFrame,
    get_somite_centroids,
    compute_anatomical_frame,
    plot_anatomical_frame,
    plot_frame_3d,
    compute_frenet_from_csv,
    plot_frenet_frame,
    project_points_to_anatomy,
    project_points_to_frenet,
    compute_hybrid_frame,
)
from whole_embryo_analysis.utils.dask_utils import setup_dask_sge_cluster, shutdown_dask
from whole_embryo_analysis.utils.viz import save_vectorized
from whole_embryo_analysis.utils.scale_resolution import to_s0, from_s0, S3_SCALE, S4_SCALE

logger = logging.getLogger(__name__)


# =============================================================================
# Coordinate Frame Building
# =============================================================================

def build_hybrid_frame(somite_mask, somite_lut, spline_csv_path,
                       pair_index_col='Somite_Stage', side_col='Side',
                       spline_smoothing=2.0,
                       spline_axis_cols=('axis-0', 'axis-1', 'axis-2')):

    somite_lut = get_somite_centroids(somite_mask, somite_lut)

    spline_df = pl.read_csv(spline_csv_path)
    spline_points = to_s0(
        np.column_stack([spline_df[c].to_numpy() for c in spline_axis_cols]),
        S4_SCALE
    )
    logger.info(f"Neural tube: {len(spline_points)} control points (s4 → s0)")

    frame = compute_hybrid_frame(
        somite_lut=somite_lut,
        spline_points=spline_points,
        pair_index_col=pair_index_col,
        side_col=side_col,
        spline_smoothing=spline_smoothing,
    )
    logger.info(f"Hybrid frame built: {len(frame.paired_numbers)} somite pairs")
    return frame


def build_anatomical_frame(somite_mask, somite_lut,
                           pair_index_col='Somite_Stage',
                           side_col='Side',
                           smoothing=5.0):

    somite_lut = get_somite_centroids(somite_mask, somite_lut)

    frame = compute_anatomical_frame(
        somite_lut,
        pair_index_col=pair_index_col,
        side_col=side_col,
        smoothing=smoothing,
    )
    logger.info(f"Anatomical frame built: {len(frame.paired_numbers)} somite pairs")
    return frame


def build_frenet_frame(spline_csv_path, smoothing=2.0):
    frame = compute_frenet_from_csv(spline_csv_path, smoothing=smoothing)
    return frame


def global_frame_visualization(frame, somite_mask, save_dir, frame_type='anatomical'):
    if isinstance(frame, FrenetFrame):
        fig = plot_frenet_frame(frame, somite_mask)
    else:
        fig = plot_anatomical_frame(frame, title='Anatomical Frame (Somite-based)')
    if save_dir is not None:
        save_vectorized(save_dir, fig, 'frame_visualization.png')
        logger.info(f"Saved frame visualization to {save_dir}")
    return fig   


def compute_biological_positions(nuclear_df, frame,
                                 z_col='Centroid_Z', y_col='Centroid_Y',
                                 x_col='Centroid_X'):
    centroids = nuclear_df.select([z_col, y_col, x_col]).to_numpy()

    if isinstance(frame, FrenetFrame):
        result = project_points_to_frenet(centroids, frame, return_frames=True)
        nuclear_df = nuclear_df.with_columns(
            pl.Series('pos_AP', result['u_param']),
            pl.Series('pos_LR', result['normal_coord']),
            pl.Series('pos_DV', result['binormal_coord']),
            pl.Series('pos_tangent', result['tangent_coord']),
        )
    else:
        result = project_points_to_anatomy(centroids, frame, return_frames=True)
        nuclear_df = nuclear_df.with_columns(
            pl.Series('pos_AP', result['pos_AP']),
            pl.Series('pos_LR', result['pos_LR']),
            pl.Series('pos_DV', result['pos_DV']),
        )

    nuc_ids = nuclear_df['Nuclear_ID'].to_list()
    local_frames = {
        int(nuc_ids[i]): result['local_frames'][i]
        for i in range(len(nuclear_df))
    }
    return nuclear_df, local_frames


# =============================================================================
# Feature Computation
# =============================================================================

def add_depth(df, somite_mask, somite_id, voxel_size=(1, 1, 1), padding=10):
    somite_id_int = int(float(somite_id))
    somite_coords = np.argwhere(somite_mask == somite_id_int)

    if len(somite_coords) == 0:
        logger.warning(f"Somite {somite_id} not found in mask")
        return df.with_columns(
            pl.lit(float('nan')).alias('depth_raw'),
            pl.lit(float('nan')).alias('depth_normalized'),
        )

    mins = np.maximum(somite_coords.min(axis=0) - padding, 0)
    maxs = np.minimum(somite_coords.max(axis=0) + 5 + padding, somite_mask.shape)

    roi = somite_mask[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
    binary_mask = (roi == somite_id_int)

    if not binary_mask.any():
        logger.warning(f"Somite {somite_id} binary mask empty after extraction")
        return df.with_columns(
            pl.lit(float('nan')).alias('depth_raw'),
            pl.lit(float('nan')).alias('depth_normalized'),
        )

    signed_dist = distance_transform_edt(binary_mask, sampling=voxel_size) \
                - distance_transform_edt(~binary_mask, sampling=voxel_size)

    centroids_s0 = df.select(['Centroid_Z', 'Centroid_Y', 'Centroid_X']).to_numpy()
    mask_indices = np.round(from_s0(centroids_s0, S3_SCALE)).astype(int)
    roi_indices = mask_indices - mins

    roi_shape = np.array(signed_dist.shape)
    roi_indices = np.clip(roi_indices, 0, roi_shape - 1)

    depths = signed_dist[roi_indices[:, 0], roi_indices[:, 1], roi_indices[:, 2]]
    max_interior = depths[depths > 0].max() if (depths > 0).any() else 1.0

    return df.with_columns(
        pl.Series('depth_raw', depths),
        pl.Series('depth_normalized', depths / max_interior),
    )


def compute_orientation_features(voxels, local_frame):
    centered = voxels - voxels.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    major_axis = eigenvectors[:, np.argmax(eigenvalues)]
    major_axis = major_axis / np.linalg.norm(major_axis)

    primary = local_frame.T @ major_axis
    if primary[0] < 0:
        primary = -primary

    align_AP, align_LR, align_DV = abs(primary[0]), abs(primary[1]), abs(primary[2])
    eig_sorted = np.sort(eigenvalues)

    return {
        'orient_AP': primary[0],
        'orient_LR': primary[1],
        'orient_DV': primary[2],
        'align_AP': align_AP,
        'align_LR': align_LR,
        'align_DV': align_DV,
        'theta_from_AP': np.arccos(np.clip(align_AP, 0, 1)),
        'phi_in_LR_DV': np.arctan2(abs(primary[2]), abs(primary[1])),
        'eccentricity': 1 - (eig_sorted[0] / eig_sorted[-1]) if eig_sorted[-1] > 0 else 0.0
    }


def process_spatial_group(group_df, nuclear_mask, local_frames):
    if group_df.is_empty():
        return {}

    zstart = int(group_df["z0"].min())
    zend = int(group_df["z1"].max())
    ystart = int(group_df["y0"].min())
    yend = int(group_df["y1"].max())
    xstart = int(group_df["x0"].min())
    xend = int(group_df["x1"].max())

    mask_roi = nuclear_mask[zstart:zend, ystart:yend, xstart:xend]
    if hasattr(mask_roi, 'compute'):
        mask_roi = mask_roi.compute()

    results = {}
    for label in group_df["label"].cast(pl.Int64).to_list():
        if label not in local_frames:
            continue

        obj_slice = (mask_roi == label).astype(np.uint8)
        if not obj_slice.any():
            continue

        vox = np.column_stack(np.where(obj_slice))
        if len(vox) < 4:
            continue

        results[label] = compute_orientation_features(vox, local_frames[label])

    return results


# =============================================================================
# Dask Cluster Management
# =============================================================================

class SomiteAnalyzer:

    def __init__(self, nuclear_mask, bbox_df, dask_params):
        self.nuclear_mask = nuclear_mask
        self.bbox_df = bbox_df
        self.dask_params = dask_params
        self.client = None
        self.cluster = None
        self.scattered_mask = None
        self._setup_attempts = 0
        self._max_setup_attempts = 3

    def _get_worker_count(self):
        try:
            if self.client is None:
                return 0
            return len(self.client.scheduler_info().get('workers', {}))
        except Exception:
            return 0

    def _check_cluster_health(self):
        count = self._get_worker_count()
        if count == 0:
            logger.warning("No active workers detected")
            return False
        return True

    def setup(self):
        self._setup_attempts += 1
        try:
            self.cluster, self.client = setup_dask_sge_cluster(
                n_workers=self.dask_params.get('num_workers'),
                cores=self.dask_params.get('cpu_cores'),
                processes=self.dask_params.get('cpu_processes'),
                memory=self.dask_params.get('cpu_memory'),
                project=self.dask_params.get('project'),
                queue=self.dask_params.get('queue'),
                runtime=self.dask_params.get('runtime'),
                resource_spec=self.dask_params.get('cpu_resource_spec'),
                log_directory=self.dask_params.get('log_dir'),
                conda_env=self.dask_params.get('conda_env'),
                dashboard_port=self.dask_params.get('dashboard_port'),
            )
            logger.info(f"Dask dashboard: {self.client.dashboard_link}")

            [self.scattered_mask] = self.client.scatter([self.nuclear_mask], broadcast=True)
            logger.info("Nuclear mask scattered to workers")
            return True
        except Exception as e:
            logger.error(f"Cluster setup failed: {e}")
            return False

    def restart_cluster(self):
        if self._setup_attempts >= self._max_setup_attempts:
            logger.error(f"Max restart attempts ({self._max_setup_attempts}) exceeded")
            return False

        logger.warning("Restarting Dask cluster...")
        self.shutdown()
        time.sleep(5)
        return self.setup()

    def ensure_cluster_ready(self):
        if not self._check_cluster_health():
            return self.restart_cluster()
        return True

    def shutdown(self):
        if self.cluster and self.client:
            try:
                shutdown_dask(self.cluster, self.client)
                logger.info("Dask cluster shut down")
            except Exception as e:
                logger.error(f"Shutdown error: {e}")
        self.client = None
        self.cluster = None
        self.scattered_mask = None

    def process_somite(self, somite_nuclear_df, local_frames):
        from dask.distributed import as_completed

        if not self.ensure_cluster_ready():
            logger.error("Cluster unavailable")
            return pl.DataFrame()

        somite_ids = set(somite_nuclear_df['Nuclear_ID'].to_list())
        somite_bbox_df = self.bbox_df.filter(pl.col('label').is_in(list(somite_ids)))

        if len(somite_bbox_df) == 0:
            return pl.DataFrame()

        try:
            groups = somite_bbox_df.partition_by(["cz", "cy", "cx"], maintain_order=True)
            [scattered_frames] = self.client.scatter([local_frames], broadcast=True)

            futures = [
                self.client.submit(process_spatial_group, gdf, self.scattered_mask, scattered_frames)
                for gdf in groups
            ]

            all_results = {}
            for batch in as_completed(futures, with_results=True).batches():
                for _, result in batch:
                    if result:
                        all_results.update(result)

            if all_results:
                records = [{'Nuclear_ID': k, **v} for k, v in all_results.items()]
                return pl.DataFrame(records)
            return pl.DataFrame()

        except Exception as e:
            logger.error(f"Error processing somite: {e}")
            if "worker" in str(e).lower():
                self.restart_cluster()
            return pl.DataFrame()
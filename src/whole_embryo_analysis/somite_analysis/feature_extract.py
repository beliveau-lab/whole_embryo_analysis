"""
Somite feature extraction pipeline.

Builds the anatomical frame once, then:
  1. Computes biological positions (pos_AP, pos_LR, pos_DV) for every nucleus
     in the whole embryo and saves ``all_nuclear_positions.csv``.
  2. Filters to somite-associated nuclei, computes per-somite depth and
     orientation features, and saves ``somite_features_{embryo_id}.csv``.

Usage:
    python -m whole_embryo_analysis.frames.feature_extract \
        --feature_path ... --outlier_path ... --somite_mask_path ... \
        --somite_lut_path ... --spline_csv_path ... \
        --nuclei_map_path ... --nuclear_mask_path ... --bbox_path ... \
        --save_dir ... --embryo_id 306
"""

import argparse
import time
import logging
from pathlib import Path
from functools import reduce

import numpy as np
import polars as pl
import tifffile
from scipy.ndimage import find_objects

import matplotlib
matplotlib.use("Agg")

from whole_embryo_analysis.somite_analysis import frames
from whole_embryo_analysis.somite_analysis.bio_frame import AnatomicalFrame
from whole_embryo_analysis.somite_analysis.plots import (
    plot_alignment_mips,
    plot_position_scatter,
)
from whole_embryo_analysis.utils.io import (
    load_n5_zarr_array,
    add_chunk_indices,
    scan_features,
    read_features,
)
from whole_embryo_analysis.utils.dask_utils import add_dask_args, dask_config_from_args

logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading
# =============================================================================

def load_all_nuclear_data(feature_path, outlier_path, id_name="Nuclear_ID"):
    """Load all nuclei (minus outliers), dropping shcoeffs columns.

    Accepts either a CSV or Parquet feature table.
    """
    q = scan_features(feature_path)
    schema = q.collect_schema()
    shcoeffs_cols = [col for col in schema.names() if col.startswith("shcoeffs")]
    outlier_ids = pl.read_csv(outlier_path)[id_name].to_list()
    df = (
        q
        .filter(~pl.col(id_name).is_in(outlier_ids))
        .drop(shcoeffs_cols)
        .collect()
    )
    logger.info(f"Loaded {len(df)} nuclei (outliers excluded)")
    return df


def join_somite_nuclei(all_nuclei_df, nuclei_map_path, id_name="Nuclear_ID"):
    """Filter to somite-associated nuclei using the nuclei map."""
    nuclei_map = pl.read_csv(nuclei_map_path).select([id_name, "Somite_ID"])
    df = all_nuclei_df.join(nuclei_map, on=id_name, how="inner")
    logger.info(f"Filtered to {len(df)} somite nuclei")
    return df


def join_pc_metadata(nuclear_df, pc_metadata, pc_cols=('PC1', 'PC2', 'PC3')):
    return nuclear_df.join(
        pc_metadata.select(['Nuclear_ID'] + list(pc_cols)),
        on='Nuclear_ID', how='left',
    )


def join_somite_metadata(nuclear_df, somite_lut):
    nuclear_dtype = nuclear_df.schema['Somite_ID']
    meta_dtype = somite_lut.schema['Somite_ID']
    if nuclear_dtype != meta_dtype:
        logger.warning(f"Somite_ID dtype mismatch: nuclear={nuclear_dtype}, lut={meta_dtype}")
        nuclear_df = nuclear_df.with_columns(pl.col('Somite_ID').cast(meta_dtype))

    result = nuclear_df.join(somite_lut, on='Somite_ID', how='inner')
    side_lower = pl.col('Side').str.to_lowercase()
    n_left = result.filter(side_lower == 'left')['Somite_Stage'].n_unique()
    n_right = result.filter(side_lower == 'right')['Somite_Stage'].n_unique()
    logger.info(f"Joined somite metadata: {n_left} left somites, {n_right} right somites")
    return result


def filter_nuclei_by_somite_id(nuclear_df, somite_id):
    dtype = nuclear_df.schema['Somite_ID']
    if dtype == pl.Utf8 or dtype == pl.String:
        match_val = str(float(somite_id))
    else:
        match_val = float(somite_id)
    return nuclear_df.filter(pl.col('Somite_ID') == match_val)


def filter_by_test_somites(nuclear_df, test_somites):
    specs = [s.strip() for s in test_somites.split(',')]
    exprs = []
    for spec in specs:
        side = spec[0].upper()
        stage = int(spec[1:])
        exprs.append(
            (pl.col('Side').str.to_uppercase().str.slice(0, 1) == side)
            & (pl.col('Somite_Stage') == stage)
        )
    combined = reduce(lambda a, b: a | b, exprs)
    filtered = nuclear_df.filter(combined)
    logger.info(f"Filtered to {specs}: {len(filtered)} nuclei")
    return filtered


# =============================================================================
# Frame Construction
# =============================================================================

def _build_frame(frame_type, args, somite_mask, somite_lut_pl):
    if frame_type == 'frenet':
        spline_csv_path = args.get('spline_csv_path')
        if not spline_csv_path:
            raise ValueError("--spline_csv_path required for frame_type='frenet'")
        logger.warning("Frenet frame: N/B axes are geometric, not anatomical")
        return frames.build_frenet_frame(spline_csv_path, smoothing=2.0)
    elif frame_type == 'hybrid':
        spline_csv_path = args.get('spline_csv_path')
        if not spline_csv_path:
            raise ValueError("--spline_csv_path required for frame_type='hybrid'")
        return frames.build_hybrid_frame(somite_mask, somite_lut_pl, spline_csv_path)
    else:
        return frames.build_anatomical_frame(somite_mask, somite_lut_pl)


# =============================================================================
# Local frame computation (for orientation features)
# =============================================================================

def compute_local_frames(somite_df, frame):
    """Return {Nuclear_ID: 3x3 local frame matrix} for each nucleus."""
    centroids = somite_df.select(['Centroid_Z', 'Centroid_Y', 'Centroid_X']).to_numpy()
    nuc_ids = somite_df['Nuclear_ID'].to_list()

    if isinstance(frame, AnatomicalFrame):
        _, nearest_idx = frame.kdt.query(centroids)
        return {
            int(nuc_ids[i]): np.column_stack([
                frame.T[nearest_idx[i]],
                frame.LR[nearest_idx[i]],
                frame.DV[nearest_idx[i]],
            ])
            for i in range(len(nuc_ids))
        }
    else:
        from whole_embryo_analysis.frames.bio_frame import project_points_to_frenet
        result = project_points_to_frenet(centroids, frame, return_frames=True)
        return {
            int(nuc_ids[i]): result['local_frames'][i]
            for i in range(len(nuc_ids))
        }


# =============================================================================
# Checkpointing
# =============================================================================

def _checkpoint_path(save_dir, csv_name):
    return Path(save_dir) / f'{csv_name}_checkpoint.csv'

def _progress_path(save_dir, csv_name):
    return Path(save_dir) / f'{csv_name}_progress.txt'

def load_checkpoint(save_dir, csv_name):
    cp = _checkpoint_path(save_dir, csv_name)
    pp = _progress_path(save_dir, csv_name)
    if not cp.exists() or not pp.exists():
        return None, set()
    try:
        checkpoint_df = pl.read_csv(cp)
        with open(pp, 'r') as f:
            completed = set(line.strip() for line in f if line.strip())
        logger.info(f"Checkpoint: {len(checkpoint_df)} nuclei, {len(completed)} somites")
        return checkpoint_df, completed
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return None, set()

def save_checkpoint(all_dfs, completed_ids, save_dir, csv_name):
    if not all_dfs:
        return
    try:
        pl.concat(all_dfs).write_csv(_checkpoint_path(save_dir, csv_name))
        with open(_progress_path(save_dir, csv_name), 'w') as f:
            f.writelines(f"{sid}\n" for sid in completed_ids)
        logger.info(f"Checkpoint: {len(completed_ids)} somites saved")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")

def cleanup_checkpoint(save_dir, csv_name):
    for path in [_checkpoint_path(save_dir, csv_name), _progress_path(save_dir, csv_name)]:
        if path.exists():
            path.unlink()
            logger.info(f"Removed {path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Somite feature extraction: biological positions + per-somite enrichment")

    parser.add_argument("--feature_path", required=True)
    parser.add_argument("--outlier_path", required=True)
    parser.add_argument("--somite_mask_path", required=True)
    parser.add_argument("--somite_lut_path", required=True)
    parser.add_argument("--nuclei_map_path", required=True,
                        help="CSV with Nuclear_ID -> Somite_ID")
    parser.add_argument("--nuclear_mask_path", required=True)
    parser.add_argument("--bbox_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--embryo_id", type=int, required=True)

    parser.add_argument("--frame_type", type=str, default="hybrid",
                        choices=["anatomical", "frenet", "hybrid"])
    parser.add_argument("--spline_csv_path", type=str, default=None)
    parser.add_argument("--positions_csv_name", type=str,
                        default="all_nuclear_positions")
    parser.add_argument("--foldback_k", type=int, default=500)
    parser.add_argument("--foldback_n_iter", type=int, default=10)
    parser.add_argument("--voxel_size", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        metavar=("Z", "Y", "X"),
                        help="Physical voxel size (µm) at somite-mask resolution "
                             "for distance_transform_edt")

    parser.add_argument("--pc_metadata_path", type=str, default=None)
    parser.add_argument("--test_somites", type=str, default=None)
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore any existing checkpoint and start from scratch")

    parser.add_argument("--camera_angles", type=float, nargs=3, default=None,
                        metavar=("ELEV", "AZIM", "ROLL"),
                        help="Napari camera angles (degrees) for position scatter plots")
    parser.add_argument("--camera_center", type=float, nargs=3, default=None,
                        metavar=("Z", "Y", "X"),
                        help="Napari camera center in s2 ZYX pixels")
    parser.add_argument("--camera_zoom", type=float, default=None,
                        help="Napari zoom level for position scatter plots")
    parser.add_argument("--camera_s_scale", type=float, default=4.0,
                        help="Downsampling factor for the camera center coords")

    add_dask_args(parser, defaults={
        "memory": "32G",
        "processes": 5,
        "runtime": "43200",
        "resource_spec": "mfree=64G",
        "conda_env": "visualization",
        "dashboard_port": ":41263",
    })

    parsed = vars(parser.parse_args())
    if parsed.get('camera_angles') is not None:
        parsed['camera_angles'] = tuple(parsed['camera_angles'])
    if parsed.get('camera_center') is not None:
        parsed['camera_center'] = tuple(parsed['camera_center'])
    return parsed


# =============================================================================
# Main
# =============================================================================

def run(args, save_visualization=True):
    dask_params = args.get('dask_params', {})
    save_dir = args['save_dir']
    embryo_id = args['embryo_id']
    frame_type = args.get('frame_type', 'hybrid')
    positions_csv_name = args.get('positions_csv_name', 'all_nuclear_positions')
    somite_csv_name = f'somite_features_{embryo_id}'
    checkpoint_interval = 5
    resume = not args.get('fresh', False)

    # -- shared data: somite mask, LUT, frame ----------------------------
    somite_mask = tifffile.imread(args['somite_mask_path'])
    somite_lut_pl = pl.read_csv(args['somite_lut_path'])
    somite_lut_pl = somite_lut_pl.filter(pl.col('Embryo_ID') == embryo_id)

    frame = _build_frame(frame_type, args, somite_mask, somite_lut_pl)
    logger.info(f"Using {frame_type} frame")

    cam_angles = args.get('camera_angles')
    cam_center = args.get('camera_center')
    cam_zoom = args.get('camera_zoom')
    cam_s_scale = args.get('camera_s_scale', 4.0)
    if cam_angles is not None:
        cam_angles = tuple(cam_angles)
    if cam_center is not None:
        cam_center = tuple(cam_center)

    if save_visualization:
        frames.global_frame_visualization(
            frame, somite_mask, save_dir, embryo_id,
            frame_type=frame_type, somite_lut=somite_lut_pl,
            camera_angles=cam_angles, camera_center=cam_center,
            camera_zoom=cam_zoom, camera_s_scale=cam_s_scale,
        )

    # =====================================================================
    # Step 1: biological positions for ALL nuclei
    # =====================================================================
    all_nuclei_df = load_all_nuclear_data(args['feature_path'], args['outlier_path'])
    logger.info(f"Loaded {len(all_nuclei_df)} total nuclei (whole embryo)")

    all_nuclei_df, all_local_frames = frames.compute_biological_positions(
        all_nuclei_df, frame,
        foldback_k=args.get('foldback_k', 500),
        foldback_n_iter=args.get('foldback_n_iter', 10),
    )
    logger.info(f"Computed biological positions for {len(all_nuclei_df)} nuclei")

    positions_df = all_nuclei_df.select([
        'Nuclear_ID', 'Centroid_Z', 'Centroid_Y', 'Centroid_X',
        'pos_AP', 'pos_LR', 'pos_DV',
    ])
    positions_path = Path(save_dir) / f'{positions_csv_name}.csv'
    positions_df.write_csv(positions_path)
    logger.info(f"Saved {len(positions_df)} positions to {positions_path}")

    if save_visualization:
        plot_position_scatter(
            positions_df, save_dir, embryo_id,
            camera_angles=cam_angles,
            camera_center=cam_center,
            camera_zoom=cam_zoom,
            camera_s_scale=cam_s_scale,
        )

    # =====================================================================
    # Step 2: per-somite enrichment (depth + orientation)
    # =====================================================================
    somite_nuclei_df = join_somite_nuclei(all_nuclei_df, args['nuclei_map_path'])
    somite_nuclei_df = join_somite_metadata(somite_nuclei_df, somite_lut_pl)

    if args.get('pc_metadata_path'):
        pc_metadata = pl.read_csv(args['pc_metadata_path'])
        somite_nuclei_df = join_pc_metadata(somite_nuclei_df, pc_metadata)

    if args.get('test_somites'):
        somite_nuclei_df = filter_by_test_somites(somite_nuclei_df, args['test_somites'])

    logger.info(f"{len(somite_nuclei_df)} somite nuclei ready for enrichment")

    nuclear_mask = load_n5_zarr_array(args['nuclear_mask_path'])
    bbox_df = read_features(args['bbox_path'])
    bbox_df = add_chunk_indices(bbox_df, nuclear_mask.chunks)

    all_somite_dfs = []
    completed_somite_ids = set()

    if resume:
        checkpoint_df, completed_somite_ids = load_checkpoint(save_dir, somite_csv_name)
        if checkpoint_df is not None:
            all_somite_dfs = [checkpoint_df]
            logger.info(f"Resuming: {len(completed_somite_ids)} somites already done")

    analyzer = frames.SomiteAnalyzer(
        args['nuclear_mask_path'], bbox_df, dask_params,
        n5_subpath=args.get('nuclear_mask_subpath'),
    )
    unique_ids = somite_nuclei_df['Somite_ID'].unique().to_list()
    to_process = [sid for sid in unique_ids if str(sid) not in completed_somite_ids]

    logger.info(f"Processing {len(to_process)} somites ({len(completed_somite_ids)} already done)")

    # Precompute per-label bounding boxes in a single pass over the somite
    # mask so add_depth() doesn't do an O(mask_volume) scan per somite.
    # find_objects returns a list where element i corresponds to label (i+1),
    # or None for labels that are absent from the mask.
    logger.info("Precomputing per-somite bounding boxes via find_objects")
    somite_slices = find_objects(somite_mask)

    # Restrict local frames to somite nuclei only, so the broadcast is small.
    somite_local_frames = {
        int(nid): all_local_frames[int(nid)]
        for nid in somite_nuclei_df['Nuclear_ID'].to_list()
        if int(nid) in all_local_frames
    }

    try:
        if not analyzer.setup():
            raise RuntimeError("Failed to setup Dask cluster")

        # One Dask sweep over every somite nucleus. Each mask chunk gets loaded
        # at most once; workers stay saturated instead of idling between somites.
        analyzer.scatter_local_frames(somite_local_frames)
        logger.info("Computing orientation features for all somite nuclei in one pass")
        t0 = time.time()
        orient_df = analyzer.process_all_somites(somite_nuclei_df)
        logger.info(
            f"Orientation features: {len(orient_df)} nuclei in {time.time() - t0:.1f}s"
        )

        # Per-somite loop now only handles depth (serial, cheap) + join.
        for i, somite_id in enumerate(to_process):
            logger.info(f"Somite {i+1}/{len(to_process)}: {somite_id}")

            somite_df = filter_nuclei_by_somite_id(somite_nuclei_df, somite_id)
            if len(somite_df) == 0:
                completed_somite_ids.add(str(somite_id))
                continue

            sid_int = int(float(somite_id))
            if 1 <= sid_int <= len(somite_slices):
                bbox_slice = somite_slices[sid_int - 1]
            else:
                bbox_slice = None
            somite_df = frames.add_depth(
                somite_df, somite_mask, somite_id,
                voxel_size=tuple(args.get('voxel_size', (1, 1, 1))),
                bbox_slice=bbox_slice,
            )

            if not orient_df.is_empty():
                somite_df = somite_df.join(orient_df, on='Nuclear_ID', how='left')
            else:
                logger.warning(f"No orientation features for somite {somite_id}")

            all_somite_dfs.append(somite_df)
            completed_somite_ids.add(str(somite_id))

            if (i + 1) % checkpoint_interval == 0:
                save_checkpoint(all_somite_dfs, completed_somite_ids, save_dir, somite_csv_name)

    except Exception as e:
        logger.error(f"Interrupted: {e}")
        save_checkpoint(all_somite_dfs, completed_somite_ids, save_dir, somite_csv_name)
        raise
    finally:
        analyzer.shutdown()

    if all_somite_dfs:
        final_df = pl.concat(all_somite_dfs)
        somite_path = Path(save_dir) / f'{somite_csv_name}.csv'
        final_df.write_csv(somite_path)
        logger.info(f"Saved {len(final_df)} somite nuclei to {somite_path}")
        cleanup_checkpoint(save_dir, somite_csv_name)

        # Alignment MIP figure (L+R pair at one stage, 3 panels)
        fig_dir = Path(save_dir) / 'figures'
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_alignment_mips(
            final_df, nuclear_mask, somite_mask, str(fig_dir),
            stage=args.get('mip_stage', 13),
        )

        return final_df
    else:
        logger.warning("No somite data to save")
        return pl.DataFrame()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    start_time = time.time()
    args = parse_args()
    args['dask_params'] = dask_config_from_args(args)
    run(args)
    logger.info(f"Feature extraction completed in {time.time() - start_time:.1f}s")

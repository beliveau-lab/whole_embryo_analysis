import argparse
import time
import logging
from pathlib import Path
from functools import reduce

import polars as pl
import tifffile

from whole_embryo_analysis.somite_analysis import somite_analysis
from whole_embryo_analysis.utils.io import load_n5_zarr_array, add_chunk_indices

logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading & Filtering
# =============================================================================

def load_nuclear_data(feature_csv, outlier_path, id_name="Nuclear_ID"):
    q = pl.scan_csv(feature_csv)
    schema = q.collect_schema()
    shcoeffs_cols = [col for col in schema.names() if col.startswith("shcoeffs")]
    outlier_ids = pl.read_csv(outlier_path)[id_name].to_list()

    return (
        q
        .filter(~pl.col(id_name).is_in(outlier_ids))
        .filter(pl.col('Somite_ID').is_not_null())
        .filter(pl.col('Somite_Side').is_not_null())
        .drop(shcoeffs_cols)
        .collect()
    )


def join_pc_metadata(nuclear_df, pc_metadata, pc_cols=('PC1', 'PC2', 'PC3')):
    pc_metadata = pc_metadata.rename({'label': 'Nuclear_ID'})
    return nuclear_df.join(
        pc_metadata.select(['Nuclear_ID'] + list(pc_cols)),
        on='Nuclear_ID', how='left',
    )


def join_somite_metadata(nuclear_df, somite_lut):
    nuclear_dtype = nuclear_df.schema['Somite_ID']
    meta_dtype = somite_lut.schema['Somite_ID']
    if nuclear_dtype != meta_dtype:
        logger.warning(f"Somite_ID dtype mismatch: nuclear={nuclear_dtype}, lut={meta_dtype}")

    result = nuclear_df.join(somite_lut, on='Somite_ID', how='inner')
    logger.info(result.head())
    n_left = result.filter(pl.col('somite_side') == 'left')['somite_stage'].n_unique()
    n_right = result.filter(pl.col('somite_side') == 'right')['somite_stage'].n_unique()
    logger.info(f"Joined somite metadata: {n_left} left somites, {n_right} right somites")
    return result


def filter_nuclei_by_somite_id(nuclear_df, somite_id):
    dtype = nuclear_df.schema['Somite_ID']
    if dtype == pl.Utf8 or dtype == pl.String:
        match_val = str(float(somite_id))
    else:
        match_val = float(somite_id)
    df = nuclear_df.filter(pl.col('Somite_ID') == match_val)
    logger.info(f"Somite_ID={match_val}: {len(df)} nuclei")
    return df


def filter_by_test_somites(nuclear_df, test_somites):
    specs = [s.strip() for s in test_somites.split(',')]

    exprs = []
    for spec in specs:
        side = spec[0].upper()
        stage = int(spec[1:])
        exprs.append(
            (pl.col('Somite_Side').str.to_uppercase().str.slice(0, 1) == side)
            & (pl.col('somite_stage') == stage)
        )

    combined = reduce(lambda a, b: a | b, exprs)
    filtered = nuclear_df.filter(combined)
    logger.info(f"Filtered to {specs}: {len(filtered)} nuclei")
    return filtered


# =============================================================================
# Configuration
# =============================================================================

def dask_config():
    return {
        "num_workers": 1,
        "cpu_memory": "32G",
        "cpu_cores": 5,
        "cpu_processes": 5,
        "cpu_resource_spec": "mfree=64G",
        "use_dask": True,
        "conda_env": "visualization",
        "project": "beliveaulab",
        "queue": "beliveau-long.q",
        "runtime": "43200",
        "log_dir": '/net/beliveau/vol1/home/msforman/msf_project/lightsheet-analysis-pipeline/logs/analysis/somites_dask/',
        "save_dir": '/net/beliveau/vol2/instrument/E9.5_306/Zoom_306/figures/',
        "dashboard_port": ":41263",
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--somite_mask_path", required=True)
    parser.add_argument("--somite_lut_path", required=True)
    parser.add_argument("--feature_path", required=True)
    parser.add_argument("--outlier_path", required=True)
    parser.add_argument("--nuclear_mask_path", required=True)
    parser.add_argument("--bbox_path", required=True)
    parser.add_argument("--save_dir", required=True)

    parser.add_argument("--pc_metadata_path", type=str, default=None)
    parser.add_argument("--frame_type", type=str, default="anatomical",
                        choices=["anatomical", "frenet", "hybrid"])
    parser.add_argument("--spline_csv_path", type=str, default=None)
    parser.add_argument("--test_somites", type=str, default=None)
    parser.add_argument("--csv_name", type=str, default='all_somite_nuclear_features')

    return vars(parser.parse_args())


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
# Frame Construction
# =============================================================================

def _build_frame(frame_type, args, somite_mask, somite_lut_pl):
    if frame_type == 'frenet':
        spline_csv_path = args.get('spline_csv_path')
        if spline_csv_path is None:
            raise ValueError("--spline_csv_path required for frame_type='frenet'")
        logger.warning("Frenet frame: N/B axes are geometric, not anatomical")
        return somite_analysis.build_frenet_frame(spline_csv_path, smoothing=2.0)

    elif frame_type == 'hybrid':
        spline_csv_path = args.get('spline_csv_path')
        if spline_csv_path is None:
            raise ValueError("--spline_csv_path required for frame_type='hybrid'")
        return somite_analysis.build_hybrid_frame(somite_mask, somite_lut_pl, spline_csv_path)

    else:
        return somite_analysis.build_anatomical_frame(somite_mask, somite_lut_pl)


# =============================================================================
# Main Analysis
# =============================================================================

def run_analysis(args, save_visualization=True):
    dask_params = args.get('dask_params', dask_config())
    resume = False
    checkpoint_interval = 5
    csv_name = args.get('csv_name', 'all_somite_nuclear_features')
    save_dir = args['save_dir']
    frame_type = args.get('frame_type', 'anatomical')

    somite_mask = tifffile.imread(args['somite_mask_path'])
    somite_lut_pl = pl.read_csv(args['somite_lut_path'])

    pc_metadata = None
    if args.get('pc_metadata_path'):
        pc_metadata = pl.read_csv(args['pc_metadata_path'])

    frame = _build_frame(frame_type, args, somite_mask, somite_lut_pl)
    logger.info(f"Using {frame_type} frame")

    if save_visualization:
        somite_analysis.global_frame_visualization(frame, somite_mask, save_dir, frame_type)

    nuclear_df_all = load_nuclear_data(args['feature_path'], args['outlier_path'])
    logger.info(f"Loaded {len(nuclear_df_all)} nuclei")

    nuclear_df_all = join_somite_metadata(nuclear_df_all, somite_lut_pl)

    if pc_metadata is not None:
        nuclear_df_all = join_pc_metadata(nuclear_df_all, pc_metadata)

    if args.get('test_somites'):
        nuclear_df_all = filter_by_test_somites(nuclear_df_all, args['test_somites'])

    nuclear_mask = load_n5_zarr_array(args['nuclear_mask_path'])
    bbox_df = pl.read_csv(args['bbox_path'])
    bbox_df = add_chunk_indices(bbox_df, nuclear_mask.chunks)

    all_somite_dfs = []
    completed_somite_ids = set()

    if resume:
        checkpoint_df, completed_somite_ids = load_checkpoint(save_dir, csv_name)
        if checkpoint_df is not None:
            all_somite_dfs = [checkpoint_df]
            logger.info(f"Resuming: {len(completed_somite_ids)} somites already done")

    analyzer = somite_analysis.SomiteAnalyzer(nuclear_mask, bbox_df, dask_params)
    unique_ids = nuclear_df_all['Somite_ID'].unique().to_list()
    to_process = [sid for sid in unique_ids if str(sid) not in completed_somite_ids]

    logger.info(f"Processing {len(to_process)} somites ({len(completed_somite_ids)} already done)")

    try:
        if not analyzer.setup():
            raise RuntimeError("Failed to setup Dask cluster")

        for i, somite_id in enumerate(to_process):
            logger.info(f"Somite {i+1}/{len(to_process)}: {somite_id}")

            somite_df = filter_nuclei_by_somite_id(nuclear_df_all, somite_id)
            if len(somite_df) == 0:
                completed_somite_ids.add(str(somite_id))
                continue

            somite_df, local_frames = somite_analysis.compute_biological_positions(somite_df, frame)
            somite_df = somite_analysis.add_depth(somite_df, somite_mask, somite_id)

            orient_df = analyzer.process_somite(somite_df, local_frames)
            if not orient_df.is_empty():
                somite_df = somite_df.join(orient_df, on='Nuclear_ID', how='left')
            else:
                logger.warning(f"No orientation features for somite {somite_id}")

            all_somite_dfs.append(somite_df)
            completed_somite_ids.add(str(somite_id))

            if (i + 1) % checkpoint_interval == 0:
                save_checkpoint(all_somite_dfs, completed_somite_ids, save_dir, csv_name)

    except Exception as e:
        logger.error(f"Interrupted: {e}")
        save_checkpoint(all_somite_dfs, completed_somite_ids, save_dir, csv_name)
        raise
    finally:
        analyzer.shutdown()

    if all_somite_dfs:
        final_df = pl.concat(all_somite_dfs)
        save_path = Path(save_dir) / f'{csv_name}.csv'
        final_df.write_csv(save_path)
        logger.info(f"Saved {len(final_df)} nuclei to {save_path}")
        cleanup_checkpoint(save_dir, csv_name)
        return final_df
    else:
        logger.warning("No data to save")
        return pl.DataFrame()


if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()
    args['dask_params'] = dask_config()
    run_analysis(args)
    logger.info(f"Completed in {time.time() - start_time:.1f}s")
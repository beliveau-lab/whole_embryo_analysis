import logging
import time
import numpy as np
import pandas as pd
import dask.array as da
import matplotlib.pyplot as plt
from aicsshparam import shtools, shparam
from dask import compute, delayed
from vtk.util import numpy_support as vtknp
import scipy.spatial as spatial
from scipy import ndimage as ndi
import zarr
from whole_embryo_analysis.utils.dask_utils import setup_dask_sge_cluster, shutdown_dask
from whole_embryo_analysis.utils.io import load_n5_zarr_array, add_chunk_indices
from whole_embryo_analysis.utils.viz import save_vectorized
from whole_embryo_analysis.utils.mesh_metrics import get_meshes_distance

import sys
import os 
import whole_embryo_analysis.align_3d as align
import argparse

DEFAULT_LMAX = 16
MIN_RELIABLE_OBJECTS = 30

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def mesh_based_distance(obj_id, mask_slice, min_volume, lmax_range, sampling_rate, alignment_method):
    obj_results = {'id': obj_id, 'distances': {}}
    label_slice = np.where(mask_slice == obj_id, mask_slice, 0)
    label_slice = np.ascontiguousarray(label_slice)
    if np.sum(label_slice > 0) < min_volume:
        logger.debug(f"Object {obj_id} is too small")
        return None
    try:
        if alignment_method == "skew":
            label_slice, _ = align.align_object_skew(label_slice, {})
        elif alignment_method == "standard":
            label_slice, _ = align.align_object_standard(label_slice, {})
        else:
            raise ValueError(f"Invalid alignment method: {alignment_method}")
        for lmax in lmax_range:
            (_, grid_rec), (_, mesh, _, _) = (
                shparam.get_shcoeffs(label_slice, lmax=lmax, alignment_2d=False)
            )
            mesh_rec = shtools.get_reconstruction_from_grid(grid_rec)
            d12, d21 = get_meshes_distance(mesh, mesh_rec)
            d12 = np.median(d12) # distance from mesh to mesh_rec
            d21 = np.median(d21) # distance from mesh_rec to mesh
            error = (d12 + d21) * sampling_rate
            if error is not None and not np.isnan(error):
                obj_results['distances'][lmax] = error 
        return obj_results
    except Exception as e:
        logger.debug(f"Failed to process object {obj_id}: {e}")
        return None
    
def process_spatial_group(group_df: pd.DataFrame,
                          mask_path: str,
                          lmax_range: list,
                          sampling_rate: float,
                          min_volume: int,
                          group_key: tuple,
                          alignment_method: str) -> list:
    if group_df.empty:
        logger.warning(f"Group {group_key}: Empty dataframe!")
        return []
    
    labels = group_df["label"].astype(np.int64).tolist()
    # Calculate bounding box that contains ALL objects in this group
    zstart = int(group_df["z0"].min()); zend = int(group_df["z1"].max())
    ystart = int(group_df["y0"].min()); yend = int(group_df["y1"].max())
    xstart = int(group_df["x0"].min()); xend = int(group_df["x1"].max())
    roi_size = (zend-zstart, yend-ystart, xend-xstart)

    logger.info(f"Group {group_key}: Processing {len(labels)} objects, ROI size: {roi_size}")
    try:
        if mask_path.endswith('.n5'):
            store = zarr.N5Store(mask_path)
            mask_arr = zarr.open_array(store=store, path=None, mode='r')
        else:
            mask_arr = zarr.open(mask_path, mode='r')
        mask_roi = mask_arr[zstart:zend, ystart:yend, xstart:xend]
        logger.info(f"Group {group_key}: Loaded ROI")
    except Exception as e:
        logger.error(f"Group {group_key}: Failed to load ROI: {e}")
        return []
    
    # Quick exit if no voxels
    if np.count_nonzero(mask_roi) == 0:
        logger.warning(f"Group {group_key}: ROI contains no masked voxels")
        return []
    
    results = []
    for i, obj_id in enumerate(labels):
        if obj_id not in np.unique(mask_roi):
            continue
        result = (
            mesh_based_distance(
                obj_id, mask_roi, min_volume, lmax_range, sampling_rate, alignment_method
            )
        )
        results.append(result)
        if (i+1) % 5 == 0:
            logger.info(f"Group {group_key}: Processed {i+1}/{len(labels)} objects")
    logger.info(f"Group {group_key}: Completed {len(results)}/{len(labels)} objects")
    return results

class ValidateLMAX:
    def __init__(self, params):
        self.params = params
        self.client = None
        self.cluster = None
        self.lmax_range = list(
            range(self.params['lmax_min'], 
                  self.params['lmax_max'], 
                  4))
        
    def workflow(self, mask_path, bboxes_path, plot_results=True, alignment_method="skew"):
        try:
            start_time = time.time()
            if self.params.get('use_dask', True):
                logger.info("Setting up Dask...")
                if not self._setup_dask():
                    raise RuntimeError(
                        "Dask cluster setup failed (e.g. log_dir may be an existing file). "
                        "Use a directory for --log_dir and ensure it does not conflict with SGE -o output file."
                    )
            self.mask_path = mask_path
            mask_array, df_bboxes = self._load_data(mask_path, bboxes_path)
            object_errors = self.run_computation(mask_array, df_bboxes, alignment_method)
            logger.info(f"Computed errors for {len(object_errors)} objects.")
            if len(object_errors) == 0:
                logger.error("No results returned! Check worker logs for errors.")
                return
            if plot_results:
                os.makedirs(self.params['save_dir'], exist_ok=True)
                self.plot_reconstruction_error(object_errors, reference_lmax=None)
            end_time = time.time()
            logger.info(f"Time taken: {end_time - start_time} seconds")
            return 
        finally:
            if self.client:
                self._shutdown_dask()

    def _load_data(self, mask_path, bboxes_path):
        """Load objects from mask. Returns DataFrame for efficient grouping."""
        logger.info("Loading data and finding objects...")
        mask_array = load_n5_zarr_array(mask_path)
        df_bboxes = pd.read_csv(bboxes_path)
        return mask_array, df_bboxes
    
    def run_computation(self, mask_array, df_bboxes, alignment_method):
        logger.info("Using optimized spatial loading strategy...")
        
        # Add spatial chunk indices 
        df_bboxes = add_chunk_indices(df_bboxes, mask_array.chunks)
        if self.params.get('min_size', 0) > 0:
            voxel_est = ((df_bboxes["z1"] - df_bboxes["z0"]) * 
                        (df_bboxes["y1"] - df_bboxes["y0"]) * 
                        (df_bboxes["x1"] - df_bboxes["x0"]))
            df_bboxes = df_bboxes.loc[voxel_est >= self.params['min_size'], :]
            logger.info(f"After size filtering: {len(df_bboxes)} objects")

        # Group by spatial chunks 
        grouped = df_bboxes.groupby(["cz", "cy", "cx"], sort=False)
        group_items = list(grouped)

        logger.info(f"Created {len(group_items)} spatial groups (avg {len(df_bboxes)/len(group_items):.1f} objects/group)")
        group_sizes = [len(gdf) for _, gdf in group_items]
        logger.info(f"Group sizes - min: {min(group_sizes)}, max: {max(group_sizes)}, mean: {np.mean(group_sizes):.1f}")

        # Process groups in parallel or sequentially
        logger.info("Processing groups with Dask distributed...")
        futures = []
        for (cz, cy, cx), group_df in group_items:
            group_df = group_df.sample(n=min(len(group_df), 5))
            fut = self.client.submit(
                process_spatial_group,
                group_df,
                self.mask_path,
                self.lmax_range,
                self.params['sampling_rate'],
                self.params['min_size'],
                (int(cz), int(cy), int(cx)),
                alignment_method
            )
            futures.append(fut)
        results = []
        for i, fut in enumerate(futures):
            try:
                group_results = fut.result()
                results.extend(group_results)
                logger.info(f"Completed {i+1}/{len(futures)} groups - got {len(group_results)} results")
            except Exception as e:
                logger.error(f"Group {i+1} processing failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info(f"Successfully processed {len(results)} objects total")
        return [r for r in results if r is not None]
    
    def structure_plotting_data(self, object_errors):
        plot_data = []
        for obj in object_errors:
            obj_id = obj['id']
            for lmax, error in obj['distances'].items():
                plot_data.append({
                    'object_id': obj_id,
                    'lmax': lmax,
                    'error': error
                })
        df = pd.DataFrame(plot_data)
        df.to_csv(f'{self.params["save_dir"]}/object_reconstruction_errors.csv', index=False)
        return df

    def plot_reconstruction_error(self, object_errors, reference_lmax=None):
        df_plot = self.structure_plotting_data(object_errors)
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        lmax_range =  list(range(4, 36, 4))
        # --- Individual Object Curves in Gray ---
        for obj_id in df_plot['object_id'].unique():
            obj_data = df_plot[df_plot['object_id'] == obj_id].sort_values('lmax')
            x_values = (obj_data['lmax'])
            ax.plot(x_values, obj_data['error'], '-', color='gray', linewidth=0.2, alpha=0.5)
        # --- Aggregated Statistics ---
        df_agg = (df_plot.groupby('lmax')['error'].agg(['mean', 'std']).reset_index().sort_values('lmax'))
        x_agg = (df_agg['lmax']) # ** 2
        ax.plot(x_agg, df_agg['mean'], '-', color='k', linewidth=2, label='mean')
        plt.axvline(x=16, linestyle='--', color='red', linewidth=2, alpha=0.5, label='chosen lmax')
        # --- Axis Formatting ---
        # ax.set_yscale('log')
        ax.set_xlim(lmax_range[0], lmax_range[-1] + 1)
        ax.set_xticks(lmax_range)
        # ax.set_xticklabels(lmax_range)
        # --- Reference Line and Title ---
        title = "Object Reconstruction Error (n="+"{:,})".format(len(df_plot))
        # --- Final Formatting ---
        ax.set_title(title)
        ax.set_xlabel('lmax', style='italic')
        ax.legend(fontsize=10)
        ax.set_ylabel('Mean distance to closest point (μm)')
        plt.tight_layout()
        save_vectorized(self.params['save_dir'], fig, 'object_reconstruction_error')
        return fig

    def _setup_dask(self):
        """Setup Dask SGE cluster for distributed processing."""
        try:
            logger.info("Setting up distributed Dask cluster...")
            self.cluster, self.client = setup_dask_sge_cluster(
                n_workers=self.params.get('num_workers'),
                cores=self.params.get('cpu_cores'),
                processes=self.params.get('cpu_processes'),
                memory=self.params.get('cpu_memory'),
                project=self.params.get('project'),
                queue=self.params.get('queue'),
                runtime=self.params.get('runtime'),
                resource_spec=self.params.get('cpu_resource_spec'),
                log_directory=self.params.get('log_dir'),
                conda_env=self.params.get('conda_env'),
                dashboard_port=self.params.get('dashboard_port')
            )
            logger.info(f"Dask dashboard link: {self.client.dashboard_link}")
            return True
        except Exception as e:
            logger.error(f"Failed to setup distributed cluster: {e}")
            return False

    def _shutdown_dask(self):
        """Shutdown Dask cluster."""
        if self.cluster and self.client:
            try:
                shutdown_dask(self.cluster, self.client)
                logger.info("Dask cluster shut down successfully")
            except Exception as e:
                logger.error(f"Error shutting down cluster: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Validate LMAX")
    parser.add_argument("--mask_path", type=str, required=True)
    parser.add_argument("--bboxes_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--plot_results", action="store_true", default=True, help="Plot reconstruction error (default True)")
    parser.add_argument("--no_plot_results", action="store_false", dest="plot_results", help="Skip plotting")
    parser.add_argument("--alignment_method", type=str, default="skew", choices=["skew", "standard"])
    return parser.parse_args()
def main():
    args = parse_args()
    params = {
        # -- dask parameters --
        "num_workers": 1,
        "cpu_memory": "512G",
        "cpu_cores": 32,
        "cpu_processes": 32,
        "cpu_resource_spec": "mfree=8G",
        # -- computation parameters --
        "sampling_rate": 2.752,
        "min_size": 4000,
        "lmax_min": 4,
        "lmax_max": 40,
        # -- tracking parameters --
        "use_dask": True,
        "conda_env": "otls-pipeline",
        "project": "beliveaulab",
        "queue": "beliveau-long.q",
        "runtime": "7200",
        "log_dir": args.log_dir,
        "save_dir": args.save_dir,
        "dashboard_port": ":41263",
    }
    ValidateLMAX(params).workflow(
        args.mask_path,
        args.bboxes_path,
        plot_results=args.plot_results,
        alignment_method=args.alignment_method,
    )

if __name__ == "__main__":
    main()

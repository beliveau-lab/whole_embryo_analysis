#!/usr/bin/env python3
"""
Surface Extraction from Segmentation Masks

This script extracts 3D surfaces from segmentation mask data stored in Zarr format.
It uses Dask for distributed processing and VTK for surface generation.
Outputs individual .vtp files for each chunk and a combined .pvd collection file.

Usage:
    python compress_surfaces.py [--zarr_path PATH] [--output_dir PATH] [--downsample N] [--help]

Environment Variables:
    ZARR_PATH: Path to input zarr file
    SURFACE_OUT_DIR: Directory for output surface files  
    LOG_DIR: Directory for log files
    DOWNSAMPLE: Downsampling factor (default: 2)
"""

import numpy as np
import dask.array as da
import dask
from dask import delayed
# from dask.distributed import Client
import zarr
import vtk
# from vtk.util import numpy_support
# import paraview.simple as pv
import logging
import os
import argparse
import sys
# import glob

# Add scripts directory to path for utils imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from whole_embryo_analysis.utils.dask_utils import setup_dask_sge_cluster, shutdown_dask


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import xml.etree.ElementTree as ET

def write_pvd_collection(vtp_paths, output_path, timestep=0):
    """
    Generate a ParaView collection (.pvd) that references many .vtp pieces.

    Parameters
    ----------
    vtp_paths : list[str]
        Paths to the per-chunk .vtp files.
    output_path : str
        Destination .pvd file.
    timestep : float or int
        Time value stored for each piece (ParaView needs one).
    """
    root = ET.Element("VTKFile", type="Collection", version="0.1")
    coll = ET.SubElement(root, "Collection")

    for path in sorted(vtp_paths):
        ET.SubElement(coll, "DataSet",
                      timestep=str(timestep),
                      group="",
                      part="0",
                      file=os.path.relpath(path, os.path.dirname(output_path)))

    tree = ET.ElementTree(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"Wrote collection file with {len(vtp_paths)} pieces → {output_path}")
    return

# Shared I/O: ensure repo root on path when run as script
import sys
from pathlib import Path
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from whole_embryo_analysis.utils.io import load_n5_zarr_array  # noqa: E402


def process_chunk_to_contours(chunk_data, chunk_offset, out_dir, block_id, spacing=(1.0, 1.0, 1.0), decimate=0.0):
    """Process single chunk to extract contour surfaces for objects.
    
    Parameters
    ----------
    chunk_data : np.ndarray
        3D array of segmentation labels for this chunk
    chunk_offset : tuple
        (z, y, x) offset of this chunk in the full array
    out_dir : str
        Output directory for .vtp files
    block_id : tuple
        Block index tuple for naming
    spacing : tuple
        (z, y, x) voxel spacing - accounts for downsampling factor
    decimate : float
        Decimation factor (0.0 = no decimation, 0.5 = reduce by 50%)
    """
    import vtk
    from vtk.util import numpy_support
    
    unique_labels = np.unique(chunk_data)
    major_labels = unique_labels[unique_labels > 0]
    
    if len(major_labels) == 0:
        return None
    
    # Ensure the output directory exists on the worker
    os.makedirs(out_dir, exist_ok=True)
    surfaces = []  # (label, vtkPolyData) pairs
    
    # Convert spacing from ZYX to XYZ for VTK
    vtk_spacing = (float(spacing[2]), float(spacing[1]), float(spacing[0]))
    vtk_origin = (
        float(chunk_offset[2]) * vtk_spacing[0],
        float(chunk_offset[1]) * vtk_spacing[1],
        float(chunk_offset[0]) * vtk_spacing[2]
    )
    
    for label in major_labels: 
        binary_data = (chunk_data == label).astype(np.uint8)
        
        # create VTK image data
        img_data = vtk.vtkImageData()
        img_data.SetDimensions(*binary_data.shape[::-1])  # VTK uses XYZ order
        img_data.SetOrigin(*vtk_origin)
        img_data.SetSpacing(*vtk_spacing)
        
        # Add data
        vtk_array = numpy_support.numpy_to_vtk(binary_data.ravel(), deep=False)
        img_data.GetPointData().SetScalars(vtk_array)
        
        # Generate contour surface
        contour = vtk.vtkContourFilter()
        contour.SetInputData(img_data)
        contour.SetValue(0, 0.5)
        contour.Update()
        
        surface = contour.GetOutput()
        if surface.GetNumberOfPoints() > 0:
            # Apply decimation if requested
            if decimate > 0.0:
                decimator = vtk.vtkDecimatePro()
                decimator.SetInputData(surface)
                decimator.SetTargetReduction(decimate)
                decimator.PreserveTopologyOn()
                decimator.Update()
                surface = decimator.GetOutput()
            
            # Generate normals for better rendering
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(surface)
            normals.ComputePointNormalsOn()
            normals.ComputeCellNormalsOff()
            normals.SplittingOff()
            normals.ConsistencyOn()
            normals.AutoOrientNormalsOn()
            normals.Update()
            surface = normals.GetOutput()
            
            # Add label as cell scalar
            label_array = vtk.vtkIntArray()
            label_array.SetName("SegmentationLabel")
            label_array.SetNumberOfTuples(surface.GetNumberOfCells())
            label_array.FillComponent(0, label)
            surface.GetCellData().SetScalars(label_array)
            
            surfaces.append((label, surface))

    if len(surfaces) == 0:
        return None

    append = vtk.vtkAppendPolyData()
    for _, surf in surfaces:
        append.AddInputData(surf)
    append.Update()

    file_name = f"block_{'_'.join(map(str, block_id))}.vtp"
    file_path = os.path.join(out_dir, file_name)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(file_path)
    writer.SetInputData(append.GetOutput())
    writer.SetDataModeToAppended()  # binary "appended-raw" section
    writer.SetCompressorTypeToZLib()
    writer.SetCompressionLevel(9)
    writer.Write()

    return [{"path": file_path, "labels": [int(l) for l, _ in surfaces]}]

def extract_surfaces(zarr_path, client, out_dir, downsample, decimate=0.0):
    """Extract surfaces from zarr in parallel chunks. Submitted to dask workers.
    
    Parameters
    ----------
    zarr_path : str
        Path to the zarr segmentation mask
    client : dask.distributed.Client
        Dask client for distributed computing
    out_dir : str
        Output directory for .vtp files
    downsample : int
        Downsampling factor (1 = no downsampling)
    decimate : float
        Mesh decimation factor (0.0 = no decimation, 0.5 = reduce by 50%)
    
    Returns
    -------
    list[dict]
        Metadata for each generated surface file
    """
    logger.info(f"Loading zarr: {zarr_path}")
    arr = load_n5_zarr_array(zarr_path)
    
    # Calculate spacing based on downsample factor
    spacing = (float(downsample), float(downsample), float(downsample))
    
    if downsample > 1:
        arr = arr[::downsample, ::downsample, ::downsample]
        logger.info(f"Applied downsampling factor: {downsample}, effective spacing: {spacing}")
    
    chunk_shape = tuple(c[0] for c in arr.chunks)
    logger.info(f"Loaded Mask (ZYX assumed): Shape={arr.shape}, Dtype={arr.dtype}, Chunk Shape: {chunk_shape}")

    delayed_tasks = []
    for block_id in np.ndindex(*arr.numblocks):
        # Calculate chunk slice and offset
        chunk_slices = tuple(
            slice(sum(arr.chunks[i][:block_id[i]]), 
                  sum(arr.chunks[i][:block_id[i]+1]))
            for i in range(arr.ndim)
        )
        chunk_offset = tuple(sum(arr.chunks[i][:block_id[i]]) for i in range(arr.ndim))
        chunk = arr[chunk_slices]
        
        # Create delayed task with spacing and decimation parameters
        task = delayed(process_chunk_to_contours)(
            chunk, chunk_offset, out_dir, block_id,
            spacing=spacing, decimate=decimate
        )
        delayed_tasks.append(task)
    
    logger.info(f"Submitting {len(delayed_tasks)} chunk processing tasks...")
    if decimate > 0:
        logger.info(f"Mesh decimation enabled: {decimate*100:.0f}% reduction target")
    
    chunk_results = client.compute(delayed_tasks, sync=True)
    metadata_list = []
    for chunk_meta in chunk_results:
        if chunk_meta:
            metadata_list.extend(chunk_meta)

    # Generate .pvd collection file
    if metadata_list:
        vtp_files = [m['path'] for m in metadata_list]
        collection_path = os.path.join(out_dir, "combined_surface.pvd")
        write_pvd_collection(vtp_files, collection_path)
    
    logger.info(f"Generated {len(metadata_list)} surface files")
    return metadata_list


def _setup_dask(params):
    """Setup Dask SGE cluster for distributed processing."""
    try:
        logger.info("Setting up distributed Dask cluster...")
        cluster, client = setup_dask_sge_cluster(
            n_workers=params.get('num_workers', 1),
            cores=params.get('cpu_cores', 4),
            processes=params.get('cpu_processes', 1),
            memory=params.get('cpu_memory', '60G'),
            project=params.get('project', 'beliveaulab'),
            queue=params.get('queue', 'beliveau-long.q'),
            runtime=params.get('runtime', '7200'),
            resource_spec=params.get('cpu_resource_spec', 'mfree=60G'),
            log_directory=params.get('log_dir', None),
            conda_env=params.get('conda_env', 'otls-pipeline'),
            dashboard_port=params.get('dashboard_port', None)
        )
        logger.info(f"Dask dashboard link: {client.dashboard_link}")
        return cluster, client
    except Exception as e:
        logger.error(f"Failed to setup distributed cluster: {e}")
        return False

def _shutdown_dask(cluster, client):
    """Shutdown Dask cluster."""
    if cluster and client:
        try:
            shutdown_dask(cluster, client)
            logger.info("Dask cluster shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down cluster: {e}")
            
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract 3D surfaces from segmentation masks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--zarr_path', type=str,
                       default=os.environ.get('ZARR_PATH', 
                               '/net/beliveau/vol2/instrument/E9.5_306/Zoom_306/dataset_fused_masks.zarr'),
                       help='Path to input zarr file')
    
    parser.add_argument('--output_dir', type=str,
                       default=os.environ.get('SURFACE_OUT_DIR',
                               '/net/beliveau/vol2/instrument/E9.5_306/Zoom_306/surfaces_compressed'),
                       help='Output directory for surface files')
    
    parser.add_argument('--log_dir', type=str,
                       default=os.environ.get('LOG_DIR',
                               '/net/beliveau/vol1/home/msforman/msf_project/lightsheet-analysis-pipeline/logs/visualization/mask_306'),
                       help='Directory for log files')
    
    parser.add_argument('--downsample', type=int,
                       default=int(os.environ.get('DOWNSAMPLE', '2')),
                       help='Downsampling factor')
    
    parser.add_argument('--num_workers', type=int, default=1,
                       help='Number of Dask workers')
    
    parser.add_argument('--cpu_memory', type=str, default='256G',
                       help='Memory per worker')
    
    parser.add_argument('--cpu_cores', type=int, default=16,
                       help='CPU cores per worker')
    
    parser.add_argument('--cpu_processes', type=int, default=32,
                       help='CPU processes per worker')
    
    parser.add_argument('--decimate', type=float, default=0.0,
                       help='Mesh decimation factor (0.0=none, 0.5=50%% reduction)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Validate input paths
    if not os.path.exists(args.zarr_path):
        logger.error(f"Input zarr file not found: {args.zarr_path}")
        sys.exit(1)
    
    # Create output directories
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Log directory: {args.log_dir}")
    except Exception as e:
        logger.error(f"Failed to create directories: {e}")
        sys.exit(1)
    
    params = {
        # -- dask parameters --
        "num_workers": args.num_workers,
        "cpu_memory": args.cpu_memory, 
        "cpu_cores": args.cpu_cores,
        "cpu_processes": args.cpu_processes,
        "cpu_resource_spec": "mfree=16G",  # RAM/worker = RAM/core * cores/worker
        # -- tracking parameters --
        'use_dask': True,
        "log_dir": args.log_dir,
        "save_dir": '/net/beliveau/vol1/home/msforman/msf_project/lightsheet-analysis-pipeline/figures/',
        "dashboard_port": ":41263",
        # -- task specific parameters --
        "zarr_path": args.zarr_path,
        "downsample": args.downsample,
        "decimate": args.decimate,
        "surface_out_dir": args.output_dir,
    }
    
    logger.info(f"Starting surface extraction with parameters:")
    logger.info(f"  Zarr path: {params['zarr_path']}")
    logger.info(f"  Output directory: {params['surface_out_dir']}")
    logger.info(f"  Downsample factor: {params['downsample']}")
    logger.info(f"  Decimation: {params['decimate']*100:.0f}%")
    logger.info(f"  Workers: {params['num_workers']}, Cores: {params['cpu_cores']}, Memory: {params['cpu_memory']}")
    
    cluster, client = _setup_dask(params)
    if not cluster or not client:
        logger.error("Failed to setup Dask cluster")
        sys.exit(1)
        
    try:
        logger.info(f"Dask dashboard: {client.dashboard_link}")
        metadata = extract_surfaces(
            params.get('zarr_path'),
            client,
            params.get('surface_out_dir'),
            downsample=params.get('downsample', 1),
            decimate=params.get('decimate', 0.0)
        )
        
        if not metadata:
            logger.warning("No surfaces were extracted")
            return
        
        collection_path = os.path.join(params["surface_out_dir"], "combined_surface.pvd")
        logger.info(
            f"Extraction complete. {len(metadata)} surface files written to {params.get('surface_out_dir')}"
        )
        logger.info(f"Collection file: {collection_path}")
        
    except Exception as e:
        logger.error(f"Error during surface extraction: {e}")
        raise
    finally:
        _shutdown_dask(cluster, client)

if __name__ == "__main__":
    main()
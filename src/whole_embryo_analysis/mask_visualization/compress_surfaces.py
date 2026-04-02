#!/usr/bin/env python3
"""
Surface Extraction from Segmentation Masks

Extracts 3D surfaces from segmentation mask data stored in Zarr format.
Uses Dask for distributed processing and VTK for surface generation.
Outputs individual .vtp files for each chunk and a combined .pvd collection file.

Usage:
    python -m whole_embryo_analysis.mask_visualization.compress_surfaces \
        --zarr_path PATH --output_dir PATH [--downsample N] [--decimate F]
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

from whole_embryo_analysis.utils.dask_utils import (
    setup_dask_sge_cluster, shutdown_dask, add_dask_args, dask_config_from_args,
)
from whole_embryo_analysis.utils.io import load_n5_zarr_array


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


def _setup_dask(dask_cfg):
    """Setup Dask SGE cluster using the standard config dict."""
    try:
        logger.info("Setting up distributed Dask cluster...")
        cluster, client = setup_dask_sge_cluster(
            n_workers=dask_cfg.get('num_workers', 1),
            cores=dask_cfg.get('cpu_cores', 4),
            processes=dask_cfg.get('cpu_processes', 1),
            memory=dask_cfg.get('cpu_memory', '60G'),
            project=dask_cfg.get('project', 'beliveaulab'),
            queue=dask_cfg.get('queue', 'beliveau-long.q'),
            runtime=dask_cfg.get('runtime', '7200'),
            resource_spec=dask_cfg.get('cpu_resource_spec', 'mfree=60G'),
            log_directory=dask_cfg.get('log_dir', None),
            conda_env=dask_cfg.get('conda_env', 'otls-pipeline'),
            dashboard_port=dask_cfg.get('dashboard_port', None)
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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--zarr_path', type=str, required=True,
                        help='Path to input zarr file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for surface files')
    parser.add_argument('--downsample', type=int, default=2,
                        help='Downsampling factor')
    parser.add_argument('--decimate', type=float, default=0.0,
                        help='Mesh decimation factor (0.0=none, 0.5=50%% reduction)')

    viz_defaults = {
        "workers": 1, "memory": "64G", "cores": 4, "processes": 1,
        "resource_spec": "mfree=64G",
    }
    add_dask_args(parser, defaults=viz_defaults)

    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.zarr_path):
        logger.error(f"Input zarr file not found: {args.zarr_path}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    dask_cfg = dask_config_from_args(args)

    logger.info("Starting surface extraction:")
    logger.info(f"  Zarr path:    {args.zarr_path}")
    logger.info(f"  Output dir:   {args.output_dir}")
    logger.info(f"  Downsample:   {args.downsample}")
    logger.info(f"  Decimation:   {args.decimate*100:.0f}%")

    cluster, client = _setup_dask(dask_cfg)
    if not cluster or not client:
        logger.error("Failed to setup Dask cluster")
        sys.exit(1)

    try:
        logger.info(f"Dask dashboard: {client.dashboard_link}")
        metadata = extract_surfaces(
            args.zarr_path,
            client,
            args.output_dir,
            downsample=args.downsample,
            decimate=args.decimate,
        )

        if not metadata:
            logger.warning("No surfaces were extracted")
            return

        collection_path = os.path.join(args.output_dir, "combined_surface.pvd")
        logger.info(f"Extraction complete. {len(metadata)} surface files → {args.output_dir}")
        logger.info(f"Collection file: {collection_path}")

    except Exception as e:
        logger.error(f"Error during surface extraction: {e}")
        raise
    finally:
        _shutdown_dask(cluster, client)

if __name__ == "__main__":
    main()
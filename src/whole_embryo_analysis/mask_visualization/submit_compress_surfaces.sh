#!/bin/bash
#$ -N compress_surfaces
#$ -l mfree=64G
#$ -pe serial 1
#$ -j y
#$ -o /net/beliveau/vol1/home/msforman/msf_project/lightsheet-analysis-pipeline/logs/visualization/compress_surfaces

PROJECT_DIR="/net/beliveau/vol1/home/msforman/msf_project/lightsheet-analysis-pipeline" # where the script is located

# Input zarr file path
ZARR_PATH='/net/beliveau/vol2/instrument/E9.5_306/Zoom_306/dataset_fused_masks.zarr'

# Output directory for surface files
SURFACE_OUT_DIR='/net/beliveau/vol2/instrument/E9.5_306/Zoom_306/surfaces_compressed'

# Log directory
LOG_DIR='/net/beliveau/vol1/home/msforman/msf_project/lightsheet-analysis-pipeline/logs/visualization/mask_306'

# Downsample factor (optional, default is 2)
DOWNSAMPLE=2

# Mesh decimation factor (0.0=none, 0.5=50% reduction) - reduces file size
DECIMATE=0.5

# Create output and log directories if they don't exist
mkdir -p "$SURFACE_OUT_DIR"
mkdir -p "$LOG_DIR"

# Activate conda environment
conda activate /net/beliveau/vol1/home/msforman/miniconda3/envs/otls-pipeline

# Change to script directory
cd /net/beliveau/vol1/home/msforman/msf_project/lightsheet-analysis-pipeline/scripts

# Set environment variables for the script
export ZARR_PATH="$ZARR_PATH"
export SURFACE_OUT_DIR="$SURFACE_OUT_DIR"
export LOG_DIR="$LOG_DIR"
export DOWNSAMPLE="$DOWNSAMPLE"

# Run the compress surfaces script
python mask_visualization/compress_surfaces.py \
    --zarr_path "$ZARR_PATH" \
    --output_dir "$SURFACE_OUT_DIR" \
    --log_dir "$LOG_DIR" \
    --downsample $DOWNSAMPLE \
    --decimate $DECIMATE

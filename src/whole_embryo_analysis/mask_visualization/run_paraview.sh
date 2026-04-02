#!/bin/bash
#$ -N paraview_job
#$ -l mfree=135G
#$ -l gpgpu=1
#$ -l cuda=1
#$ -pe serial 2
#$ -j y
#$ -o /net/beliveau/vol1/home/msforman/msf_project/lightsheet-analysis-pipeline/logs/logs_dir/zoom_306

# Define paths
PARAVIEW_BIN_DIR="/net/beliveau/vol1/home/msforman/ParaView-6.0.0-RC1-MPI-Linux-Python3.12-x86_64/bin/"
PROJECT_DIR="/net/beliveau/vol1/home/msforman/msf_project/lightsheet-analysis-pipeline"
SCRIPTS_DIR="$PROJECT_DIR/scripts/mask_visualization"
OUTPUT_DIR="/net/beliveau/vol1/home/msforman/msf_project/lightsheet-analysis-pipeline/figures/vtp_stuff/"
SURFACE_DIR='/net/beliveau/vol2/instrument/E9.5_306/Zoom_306/surfaces_compressed'
OUTPUT_TAG='_'

EMBRYONIC_NUMBER=306

export PYTHONUNBUFFERED=1

cd $PARAVIEW_BIN_DIR
mpirun -np $NSLOTS pvbatch $SCRIPTS_DIR/render_segmentations.py --embryonic_number $EMBRYONIC_NUMBER --output_dir $OUTPUT_DIR --surface_dir $SURFACE_DIR --num_frames 25 --output_tag $OUTPUT_TAG
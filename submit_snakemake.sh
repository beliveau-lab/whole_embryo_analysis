#!/bin/bash
#$ -N analysis
#$ -l mfree=64G
#$ -pe serial 1
#$ -j y
#$ -o /net/beliveau/vol1/home/msforman/msf_project/whole_embryo_analysis/logs/snakemake.log
#$ -V

conda activate /net/beliveau/vol1/home/msforman/.conda/envs/visualization
cd /net/beliveau/vol1/home/msforman/msf_project/whole_embryo_analysis/

which snakemake

snakemake --unlock || true

snakemake -p --configfile config/config.yaml --cores 1 "$@"
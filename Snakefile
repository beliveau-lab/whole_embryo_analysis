"""
whole_embryo_analysis Snakemake workflow
========================================
Three-stage pipeline:
  1. kde_outliers     — KDE-based outlier detection on nuclear shape metrics
  2. shapespace_pca   — PCA on spherical harmonic coefficients
  3. somite_features  — Per-nucleus somite feature extraction
"""

configfile: "config/config.yaml"

OUTPUT_DIR = config["output_dir"]

# Derived output paths
KDE_DIR        = f"{OUTPUT_DIR}/KDE"
OUTLIER_CSV    = f"{KDE_DIR}/outliers.csv"
KDE_FIG_DIR    = f"{KDE_DIR}/figures"

SHP_TAG        = config["shapespace"]["str_tag"]
SHAPESPACE_CSV = f"{OUTPUT_DIR}/Shapemodes_{SHP_TAG}/{SHP_TAG}_shapespace.csv"

SOMITE_CSV     = f"{OUTPUT_DIR}/somite_classification_data/{config['somite']['csv_name']}.csv"


# ── default target ───────────────────────────────────────────────────────────

rule all:
    input:
        SOMITE_CSV


# ── 1. KDE outlier detection ────────────────────────────────────────────────

rule kde_outliers:
    input:
        feature_csv = config["feature_csv"],
    output:
        outlier_csv = OUTLIER_CSV,
        figure_dir  = directory(KDE_FIG_DIR),
    params:
        density_thresh = config["kde"]["density_thresh"],
        n_subsample    = config["kde"]["n_subsample"],
        n_rounds       = config["kde"]["n_rounds"],
        dask           = config["kde"]["dask"],
    log:
        "logs/kde_outliers.log"
    shell:
        """
        python -m whole_embryo_analysis.find_outliers \
            --csv_path {input.feature_csv} \
            --figure_dir {output.figure_dir} \
            --outlier_path {output.outlier_csv} \
            --density_thresh {params.density_thresh} \
            --n_subsample {params.n_subsample} \
            --n_rounds {params.n_rounds} \
            --dask_workers {params.dask[num_workers]} \
            --dask_memory {params.dask[memory]} \
            --dask_cores {params.dask[cores]} \
            --dask_processes {params.dask[processes]} \
            --dask_project {params.dask[project]} \
            --dask_queue {params.dask[queue]} \
            --dask_runtime {params.dask[runtime]} \
            --dask_resource_spec {params.dask[resource_spec]} \
            --dask_conda_env {params.dask[conda_env]} \
            --dask_dashboard_port {params.dask[dashboard_port]} \
            --dask_log_dir {params.dask[log_dir]} \
            2>&1 | tee {log}
        """


# ── 2. Shapespace PCA ───────────────────────────────────────────────────────

rule shapespace_pca:
    input:
        feature_csv = config["feature_csv"],
        outlier_csv = OUTLIER_CSV,
    output:
        shapespace_csv = SHAPESPACE_CSV,
    params:
        str_tag          = config["shapespace"]["str_tag"],
        id_name          = config["shapespace"]["id_name"],
        lmax             = config["shapespace"]["lmax"],
        num_pcs          = config["shapespace"]["num_pcs"],
        store_flag       = "--store_results" if config["shapespace"]["store_results"] else "",
        figures_flag     = "--generate_figures" if config["shapespace"]["generate_figures"] else "",
    log:
        "logs/shapespace_pca.log"
    shell:
        """
        python -m whole_embryo_analysis.shapespace.main \
            --feature_csv {input.feature_csv} \
            --outlier_path {input.outlier_csv} \
            --save_dir {OUTPUT_DIR} \
            --str_tag {params.str_tag} \
            --id_name {params.id_name} \
            --lmax {params.lmax} \
            --num_pcs {params.num_pcs} \
            {params.store_flag} \
            {params.figures_flag} \
            2>&1 | tee {log}
        """


# ── 3. Somite feature extraction ────────────────────────────────────────────

rule somite_features:
    input:
        feature_csv    = config["feature_csv"],
        outlier_csv    = OUTLIER_CSV,
        shapespace_csv = SHAPESPACE_CSV,
        somite_mask    = config["somite_mask"],
        somite_lut     = config["somite_lut"],
        nuclear_mask   = config["nuclear_mask"],
        bbox_csv       = config["bbox_csv"],
    output:
        somite_csv = SOMITE_CSV,
    params:
        frame_type = config["somite"]["frame_type"],
        csv_name   = config["somite"]["csv_name"],
        dask       = config["somite"]["dask"],
        embryo_id  = config["embryo_id"],
    log:
        "logs/somite_features.log"
    shell:
        """
        python -m whole_embryo_analysis.somite_analysis.feature_extract \
            --feature_path {input.feature_csv} \
            --outlier_path {input.outlier_csv} \
            --pc_metadata_path {input.shapespace_csv} \
            --somite_mask_path {input.somite_mask} \
            --somite_lut_path {input.somite_lut} \
            --nuclear_mask_path {input.nuclear_mask} \
            --bbox_path {input.bbox_csv} \
            --save_dir $(dirname {output.somite_csv}) \
            --frame_type {params.frame_type} \
            --csv_name {params.csv_name} \
            --embryo_id {params.embryo_id} \
            --dask_workers {params.dask[num_workers]} \
            --dask_memory {params.dask[memory]} \
            --dask_cores {params.dask[cores]} \
            --dask_processes {params.dask[processes]} \
            --dask_project {params.dask[project]} \
            --dask_queue {params.dask[queue]} \
            --dask_runtime {params.dask[runtime]} \
            --dask_resource_spec {params.dask[resource_spec]} \
            --dask_conda_env {params.dask[conda_env]} \
            --dask_dashboard_port {params.dask[dashboard_port]} \
            --dask_log_dir {params.dask[log_dir]} \
            2>&1 | tee {log}
        """

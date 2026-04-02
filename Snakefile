"""
whole_embryo_analysis Snakemake workflow
========================================
Five-stage pipeline:
  1. kde_outliers            — KDE-based outlier detection on nuclear shape metrics
  2. shapespace_pca          — PCA on spherical harmonic coefficients
  3. somite_features         — Per-nucleus somite feature extraction
  4. compress_surfaces       — Extract 3D surfaces from segmentation masks (Dask)
  5. render_segmentations    — Render surfaces with ParaView (MPI + GPU)
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

SURFACE_DIR    = config["visualization"]["surface_dir"]
PVD_FILE       = f"{SURFACE_DIR}/combined_surface.pvd"
RENDER_DIR     = config["visualization"]["render_output_dir"]
RENDER_DONE    = f"{RENDER_DIR}/render.done"


# ── default target ───────────────────────────────────────────────────────────
rule all:
    input:
        SOMITE_CSV,
        RENDER_DONE,

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


# ── 4. Compress surfaces (Dask) ─────────────────────────────────────────────

rule compress_surfaces:
    input:
        segmentation_mask = config["visualization"]["segmentation_mask"],
        somite_csv        = SOMITE_CSV,
    output:
        pvd_file = PVD_FILE,
    params:
        surface_dir = SURFACE_DIR,
        downsample  = config["visualization"]["downsample"],
        decimate    = config["visualization"]["decimate"],
        dask        = config["visualization"]["dask"],
    log:
        "logs/compress_surfaces.log"
    shell:
        """
        python -m whole_embryo_analysis.mask_visualization.compress_surfaces \
            --zarr_path {input.segmentation_mask} \
            --output_dir {params.surface_dir} \
            --downsample {params.downsample} \
            --decimate {params.decimate} \
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


# ── 5. Render segmentations (ParaView pvbatch + MPI) ────────────────────────

rule render_segmentations:
    input:
        pvd_file = PVD_FILE,
    output:
        done = touch(RENDER_DONE),
    params:
        paraview_bin = config["visualization"]["paraview_bin_dir"],
        surface_dir  = SURFACE_DIR,
        render_dir   = RENDER_DIR,
        embryo_id    = config["embryo_id"],
        num_frames   = config["visualization"]["num_frames"],
        output_tag   = config["visualization"]["output_tag"],
        mpi_slots    = config["visualization"]["mpi_slots"],
        script       = srcdir("src/whole_embryo_analysis/mask_visualization/render_segmentations.py"),
    log:
        "logs/render_segmentations.log"
    shell:
        """
        cd {params.paraview_bin} && \
        mpirun -np {params.mpi_slots} pvbatch {params.script} \
            --embryonic_number {params.embryo_id} \
            --output_dir {params.render_dir} \
            --surface_dir {params.surface_dir} \
            --num_frames {params.num_frames} \
            --output_tag {params.output_tag} \
            2>&1 | tee {log}
        """

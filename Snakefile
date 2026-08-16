"""
whole_embryo_analysis Snakemake workflow
========================================
Seven-stage pipeline:
  1. find_outliers            — KDE-based outlier detection on nuclear shape metrics
  2. shapespace_pca          — PCA on spherical harmonic coefficients
  3. somite_nuclei_map       — Assign Somite_ID to each nucleus via mask lookup
  4. somite_features         — Biological positions + per-somite depth & orientation
  5. somite_clustering       — Hierarchical clustering + L-R enrichment figures
  6. compress_surfaces       — Extract 3D surfaces from segmentation masks (Dask)
  7. render_segmentations    — Render surfaces with ParaView (MPI + GPU)
"""

import os
import time as _time
_pipeline_start = _time.time()

# No configfile: directive — must be supplied on the command line
if not config:
    raise WorkflowError(
        "No config provided. Run with: snakemake --configfile configs/<run>.yaml"
    )

onsuccess:
    elapsed = _time.time() - _pipeline_start
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print(f"Pipeline completed successfully in {h}h {m}m {s}s")

onerror:
    elapsed = _time.time() - _pipeline_start
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print(f"Pipeline failed after {h}h {m}m {s}s -- check log files in {LOG_DIR}/")

# --- Derive input paths from sample_dir + embryo_id --------------------------
SAMPLE_DIR = config["sample_dir"]
EMBRYO_ID  = str(config["embryo_id"])

FEATURE_CSV  = config.get("feature_csv",
    f"{SAMPLE_DIR}/dataset_fused_features-JGseg-030926_align_skew.csv")
BBOX_CSV     = config.get("bbox_csv",
    f"{SAMPLE_DIR}/dataset_fused_features-JGseg-030926_align_skew_bboxes.csv")
NUCLEAR_MASK = config.get("nuclear_mask",
    f"{SAMPLE_DIR}/dataset_fused_masks_Julia_FT.zarr")
SOMITE_MASK  = config.get("somite_mask",
    f"{SAMPLE_DIR}/exported_tiff_for_somite_mask_correction/image_s3_{EMBRYO_ID}_cp_masks_corrected.tif")
SPLINE_CSV   = config.get("somite", {}).get("spline_csv_path", "")

OUTPUT_DIR = config.get("output_dir", f"{SAMPLE_DIR}/analysis_pipeline")

# Log namespacing: prefer explicit run_id, fall back to embryo_id.
# Absolute so that rules which `cd` elsewhere (render_segmentations) still
# write their logs into the project tree.
RUN_ID  = str(config.get("run_id", EMBRYO_ID))
LOG_DIR = os.path.abspath(config.get("log_dir", f"logs/{RUN_ID}"))

# Derived output paths
FEATURE_DIR     = f"{OUTPUT_DIR}/parquet_files"
FEATURE_PARQUET = f"{FEATURE_DIR}/nuclear_features.parquet"
BBOX_PARQUET    = f"{FEATURE_DIR}/nuclear_bboxes.parquet"

KDE_DIR        = f"{OUTPUT_DIR}/outlier_detection"
OUTLIER_CSV    = f"{KDE_DIR}/outliers.csv"
KDE_FIG_DIR    = f"{KDE_DIR}/figures"

SHP_TAG        = config["shapespace"]["str_tag"]
SHAPESPACE_CSV = f"{OUTPUT_DIR}/shapemodes_{SHP_TAG}/{SHP_TAG}_shapespace.csv"

SOMITE_DIR     = f"{OUTPUT_DIR}/somite_data"
NUCLEI_MAP_CSV = f"{SOMITE_DIR}/{EMBRYO_ID}_somite_nuclei_map.csv"

POSITIONS_CSV  = f"{SOMITE_DIR}/biological_coordinates.csv"
SOMITE_CSV     = f"{SOMITE_DIR}/somite_features_{EMBRYO_ID}.csv"

CLUSTER_CSV    = f"{SOMITE_DIR}/somite_clustered_{EMBRYO_ID}.csv"

HAS_VIZ = "visualization" in config
if HAS_VIZ:
    SEG_MASK    = config["visualization"].get("segmentation_mask",
        f"{SAMPLE_DIR}/dataset_fused_masks_Julia_FT.zarr")
    SURFACE_DIR = config["visualization"].get("surface_dir",
        f"{SAMPLE_DIR}/surfaces_compressed")
    PVD_FILE    = f"{SURFACE_DIR}/combined_surface.pvd"
    RENDER_DIR  = config["visualization"].get("render_output_dir",
        f"{OUTPUT_DIR}/rendered_frames")
    RENDER_DONE = f"{RENDER_DIR}/render.done"


# ── default target ───────────────────────────────────────────────────────────
rule all:
    input:
        CLUSTER_CSV,
        *([RENDER_DONE] if HAS_VIZ else []),

# ── 0. Convert feature CSV + bbox CSV → Parquet ─────────────────────────────
# Columnar! Woohoo

rule features_to_parquet:
    input:
        feature_csv = FEATURE_CSV,
        bbox_csv    = BBOX_CSV,
    output:
        feature_parquet = FEATURE_PARQUET,
        bbox_parquet    = BBOX_PARQUET,
    params:
        script = srcdir("scripts/csv_to_parquet.py"),
    log:
        f"{LOG_DIR}/features_to_parquet.log"
    shell:
        """
        python -m whole_embryo_analysis.to_parquet \
            --pair {input.feature_csv} {output.feature_parquet} \
            --pair {input.bbox_csv} {output.bbox_parquet} \
            --compression zstd \
            2>&1 | tee {log}
        """

# ── 1. Outlier detection (Mahalanobis) ─────────────────────

rule find_outliers:
    input:
        feature_parquet = FEATURE_PARQUET,
    output:
        outlier_csv = OUTLIER_CSV,
        figure_dir  = directory(KDE_FIG_DIR),
    params:
        alpha       = config["outlier_detection"].get("alpha", 0.001),
        qc_mask     = NUCLEAR_MASK,
        qc_image    = config.get("qc_image_path", f"{SAMPLE_DIR}/dataset_fused_rechunked_blocks.zarr"),
        qc_channel  = config.get("qc_channel", 2),
        qc_n        = config.get("qc_n", 16),
    log:
        f"{LOG_DIR}/find_outliers.log"
    shell:
        """
        python -m whole_embryo_analysis.find_outliers \
            --csv_path {input.feature_parquet} \
            --figure_dir {output.figure_dir} \
            --outlier_path {output.outlier_csv} \
            --alpha {params.alpha} \
            --qc_nuclear_mask {params.qc_mask} \
            --qc_image_path {params.qc_image} \
            --qc_channel {params.qc_channel} \
            --qc_n {params.qc_n} \
            2>&1 | tee {log}
        """


# ── 2. Shapespace PCA ───────────────────────────────────────────────────────

rule shapespace_pca:
    input:
        feature_parquet = FEATURE_PARQUET,
        outlier_csv     = OUTLIER_CSV,
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
        f"{LOG_DIR}/shapespace_pca.log"
    shell:
        """
        python -m whole_embryo_analysis.shapespace.main \
            --feature_csv {input.feature_parquet} \
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


# ── 3. Somite-nucleus mapping ────────────────────────────────────────────────

rule somite_nuclei_map:
    input:
        feature_parquet = FEATURE_PARQUET,
        somite_mask     = SOMITE_MASK,
        somite_lut      = config["somite_lut"],
    output:
        nuclei_map = NUCLEI_MAP_CSV,
    params:
        output_dir = f"{OUTPUT_DIR}/somite_data",
        sample_id  = EMBRYO_ID,
    log:
        f"{LOG_DIR}/somite_nuclei_map.log"
    shell:
        """
        python -m whole_embryo_analysis.somite_nuclei_map \
            --nuclear_features_path {input.feature_parquet} \
            --somite_mask_path {input.somite_mask} \
            --somite_lut_path {input.somite_lut} \
            --output_path {params.output_dir} \
            --sample_id {params.sample_id} \
            2>&1 | tee {log}
        """


# ── 4. Somite feature extraction ────────────────────────────────────────────

rule somite_features:
    input:
        feature_parquet = FEATURE_PARQUET,
        outlier_csv     = OUTLIER_CSV,
        shapespace_csv  = SHAPESPACE_CSV,
        nuclei_map      = NUCLEI_MAP_CSV,
        somite_mask     = SOMITE_MASK,
        somite_lut      = config["somite_lut"],
        nuclear_mask    = NUCLEAR_MASK,
        bbox_parquet    = BBOX_PARQUET,
    output:
        positions_csv = POSITIONS_CSV,
        somite_csv    = SOMITE_CSV,
    params:
        save_dir          = SOMITE_DIR,
        embryo_id         = config["embryo_id"],
        frame_type        = "hybrid",
        spline_csv_path   = SPLINE_CSV,
        positions_csv_name = config["somite"]["positions_csv_name"],
        foldback_k        = config["somite"].get("foldback_k", 500),
        foldback_n_iter   = config["somite"].get("foldback_n_iter", 10),
        voxel_size        = config["somite"].get("voxel_size", [1.0, 1.0, 1.0]),
        camera_angles     = config["somite"].get("camera_angles", []),
        camera_center     = config["somite"].get("camera_center", []),
        camera_zoom       = config["somite"].get("camera_zoom", ""),
        camera_s_scale    = config["somite"].get("camera_s_scale", 4.0),
        dask              = config["somite"]["dask"],
    log:
        f"{LOG_DIR}/somite_features.log"
    shell:
        """
        python -m whole_embryo_analysis.somite_analysis.feature_extract \
            --feature_path {input.feature_parquet} \
            --outlier_path {input.outlier_csv} \
            --nuclei_map_path {input.nuclei_map} \
            --pc_metadata_path {input.shapespace_csv} \
            --somite_mask_path {input.somite_mask} \
            --somite_lut_path {input.somite_lut} \
            --nuclear_mask_path {input.nuclear_mask} \
            --bbox_path {input.bbox_parquet} \
            --save_dir {params.save_dir} \
            --embryo_id {params.embryo_id} \
            --frame_type {params.frame_type} \
            --spline_csv_path {params.spline_csv_path} \
            --positions_csv_name {params.positions_csv_name} \
            --foldback_k {params.foldback_k} \
            --foldback_n_iter {params.foldback_n_iter} \
            --voxel_size {params.voxel_size[0]} {params.voxel_size[1]} {params.voxel_size[2]} \
            --camera_angles {params.camera_angles[0]} {params.camera_angles[1]} {params.camera_angles[2]} \
            --camera_center {params.camera_center[0]} {params.camera_center[1]} {params.camera_center[2]} \
            --camera_zoom {params.camera_zoom} \
            --camera_s_scale {params.camera_s_scale} \
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


# ── 5. Somite clustering ────────────────────────────────────────────────────

rule somite_clustering:
    input:
        somite_csv   = SOMITE_CSV,
        nuclear_mask = NUCLEAR_MASK,
        somite_mask  = SOMITE_MASK,
    output:
        cluster_csv = CLUSTER_CSV,
    params:
        save_dir    = SOMITE_DIR,
        embryo_id   = config["embryo_id"],
        n_clusters  = config["clustering"]["n_clusters"],
        mip_stage   = 13,
    log:
        f"{LOG_DIR}/somite_clustering.log"
    shell:
        """
        python -m whole_embryo_analysis.somite_analysis.somite_cluster \
            --somite_csv {input.somite_csv} \
            --save_dir {params.save_dir} \
            --embryo_id {params.embryo_id} \
            --n_clusters {params.n_clusters} \
            --nuclear_mask_path {input.nuclear_mask} \
            --somite_mask_path {input.somite_mask} \
            --mip_stage {params.mip_stage} \
            2>&1 | tee {log}
        """


# ── 6. Compress surfaces (Dask) ─────────────────────────────────────────────

if HAS_VIZ:
    rule compress_surfaces:
        input:
            segmentation_mask = SEG_MASK,
            somite_csv        = SOMITE_CSV,
        output:
            pvd_file = PVD_FILE,
        params:
            surface_dir       = SURFACE_DIR,
            downsample        = config["visualization"]["downsample"],
            decimate          = config["visualization"]["decimate"],
            mask_subpath_flag = (
                f"--mask_subpath {config['visualization']['mask_subpath']}"
                if config["visualization"].get("mask_subpath") else ""
            ),
            dask              = config["visualization"]["dask"],
        log:
            f"{LOG_DIR}/compress_surfaces.log"
        shell:
            """
            python -m whole_embryo_analysis.mask_visualization.compress_surfaces \
                --zarr_path {input.segmentation_mask} \
                --output_dir {params.surface_dir} \
                --downsample {params.downsample} \
                --decimate {params.decimate} \
                {params.mask_subpath_flag} \
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


    # ── 7. Render segmentations (ParaView pvbatch + MPI) ─────────────────────

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
            f"{LOG_DIR}/render_segmentations.log"
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


# """
# whole_embryo_analysis Snakemake workflow
# ========================================
# Seven-stage pipeline:
#   1. find_outliers            — KDE-based outlier detection on nuclear shape metrics
#   2. shapespace_pca          — PCA on spherical harmonic coefficients
#   3. somite_nuclei_map       — Assign Somite_ID to each nucleus via mask lookup
#   4. somite_features         — Biological positions + per-somite depth & orientation
#   5. somite_clustering       — Hierarchical clustering + L-R enrichment figures
#   6. compress_surfaces       — Extract 3D surfaces from segmentation masks (Dask)
#   7. render_segmentations    — Render surfaces with ParaView (MPI + GPU)
# """

# import time as _time
# _pipeline_start = _time.time()

# # No configfile: directive — must be supplied on the command line
# if not config:
#     raise WorkflowError(
#         "No config provided. Run with: snakemake --configfile configs/<run>.yaml"
#     )

# onsuccess:
#     elapsed = _time.time() - _pipeline_start
#     h, rem = divmod(int(elapsed), 3600)
#     m, s = divmod(rem, 60)
#     print(f"Pipeline completed successfully in {h}h {m}m {s}s")

# onerror:
#     elapsed = _time.time() - _pipeline_start
#     h, rem = divmod(int(elapsed), 3600)
#     m, s = divmod(rem, 60)
#     print(f"Pipeline failed after {h}h {m}m {s}s -- check log files in logs/")

# # --- Derive input paths from sample_dir + embryo_id --------------------------
# SAMPLE_DIR = config["sample_dir"]
# EMBRYO_ID  = str(config["embryo_id"])

# FEATURE_CSV  = config.get("feature_csv",
#     f"{SAMPLE_DIR}/dataset_fused_features-JGseg-030926_align_skew.csv")
# BBOX_CSV     = config.get("bbox_csv",
#     f"{SAMPLE_DIR}/dataset_fused_features-JGseg-030926_align_skew_bboxes.csv")
# NUCLEAR_MASK = config.get("nuclear_mask",
#     f"{SAMPLE_DIR}/dataset_fused_masks_Julia_FT.zarr")
# SOMITE_MASK  = config.get("somite_mask",
#     f"{SAMPLE_DIR}/exported_tiff_for_somite_mask_correction/image_s3_{EMBRYO_ID}_cp_masks_corrected.tif")
# SPLINE_CSV   = config.get("somite", {}).get("spline_csv_path", "")

# OUTPUT_DIR = config.get("output_dir", f"{SAMPLE_DIR}/analysis_pipeline")

# # Derived output paths
# FEATURE_DIR     = f"{OUTPUT_DIR}/parquet_files"
# FEATURE_PARQUET = f"{FEATURE_DIR}/nuclear_features.parquet"
# BBOX_PARQUET    = f"{FEATURE_DIR}/nuclear_bboxes.parquet"

# KDE_DIR        = f"{OUTPUT_DIR}/outlier_detection"
# OUTLIER_CSV    = f"{KDE_DIR}/outliers.csv"
# KDE_FIG_DIR    = f"{KDE_DIR}/figures"

# SHP_TAG        = config["shapespace"]["str_tag"]
# SHAPESPACE_CSV = f"{OUTPUT_DIR}/shapemodes_{SHP_TAG}/{SHP_TAG}_shapespace.csv"

# SOMITE_DIR     = f"{OUTPUT_DIR}/somite_data"
# NUCLEI_MAP_CSV = f"{SOMITE_DIR}/{EMBRYO_ID}_somite_nuclei_map.csv"

# POSITIONS_CSV  = f"{SOMITE_DIR}/biological_coordinates.csv"
# SOMITE_CSV     = f"{SOMITE_DIR}/somite_features_{EMBRYO_ID}.csv"

# CLUSTER_CSV    = f"{SOMITE_DIR}/somite_clustered_{EMBRYO_ID}.csv"

# HAS_VIZ = "visualization" in config
# if HAS_VIZ:
#     SEG_MASK    = config["visualization"].get("segmentation_mask",
#         f"{SAMPLE_DIR}/dataset_fused_masks_Julia_FT.zarr")
#     SURFACE_DIR = config["visualization"].get("surface_dir",
#         f"{SAMPLE_DIR}/surfaces_compressed")
#     PVD_FILE    = f"{SURFACE_DIR}/combined_surface.pvd"
#     RENDER_DIR  = config["visualization"].get("render_output_dir",
#         f"{OUTPUT_DIR}/rendered_frames")
#     RENDER_DONE = f"{RENDER_DIR}/render.done"


# # ── default target ───────────────────────────────────────────────────────────
# rule all:
#     input:
#         CLUSTER_CSV,
#         *([RENDER_DONE] if HAS_VIZ else []),

# # ── 0. Convert feature CSV + bbox CSV → Parquet ─────────────────────────────
# # One-shot streaming conversion. Downstream rules consume Parquet so we avoid
# # re-parsing the multi-GB CSVs on every step (huge win for 3TB-scale samples).

# rule features_to_parquet:
#     input:
#         feature_csv = FEATURE_CSV,
#         bbox_csv    = BBOX_CSV,
#     output:
#         feature_parquet = FEATURE_PARQUET,
#         bbox_parquet    = BBOX_PARQUET,
#     log:
#         "logs/features_to_parquet.log"
#     shell:
#         """
#         python -c "
# import polars as pl, sys, os
# os.makedirs(os.path.dirname('{output.feature_parquet}'), exist_ok=True)
# for src, dst in [('{input.feature_csv}', '{output.feature_parquet}'),
#                  ('{input.bbox_csv}', '{output.bbox_parquet}')]:
#     print(f'Converting {{src}} -> {{dst}}')
#     lf = pl.scan_csv(src, infer_schema_length=10000)
#     try:
#         lf.sink_parquet(dst)
#     except Exception as e:
#         print(f'sink_parquet failed ({{e}}), falling back to collect()')
#         lf.collect(streaming=True).write_parquet(dst)
# " 2>&1 | tee {log}
#         """

# # ── 1. Outlier detection (morphological + Mahalanobis) ─────────────────────

# rule find_outliers:
#     input:
#         feature_parquet = FEATURE_PARQUET,
#     output:
#         outlier_csv = OUTLIER_CSV,
#         figure_dir  = directory(KDE_FIG_DIR),
#     params:
#         alpha       = config["outlier_detection"].get("alpha", 0.001),
#         qc_mask     = NUCLEAR_MASK,
#         qc_image    = config.get("qc_image_path", f"{SAMPLE_DIR}/dataset_fused_rechunked_blocks.zarr"),
#         qc_channel  = config.get("qc_channel", 2),
#         qc_n        = config.get("qc_n", 16),
#     log:
#         "logs/find_outliers.log"
#     shell:
#         """
#         python -m whole_embryo_analysis.find_outliers \
#             --csv_path {input.feature_parquet} \
#             --figure_dir {output.figure_dir} \
#             --outlier_path {output.outlier_csv} \
#             --alpha {params.alpha} \
#             --qc_nuclear_mask {params.qc_mask} \
#             --qc_image_path {params.qc_image} \
#             --qc_channel {params.qc_channel} \
#             --qc_n {params.qc_n} \
#             2>&1 | tee {log}
#         """


# # ── 2. Shapespace PCA ───────────────────────────────────────────────────────

# rule shapespace_pca:
#     input:
#         feature_parquet = FEATURE_PARQUET,
#         outlier_csv     = OUTLIER_CSV,
#     output:
#         shapespace_csv = SHAPESPACE_CSV,
#     params:
#         str_tag          = config["shapespace"]["str_tag"],
#         id_name          = config["shapespace"]["id_name"],
#         lmax             = config["shapespace"]["lmax"],
#         num_pcs          = config["shapespace"]["num_pcs"],
#         store_flag       = "--store_results" if config["shapespace"]["store_results"] else "",
#         figures_flag     = "--generate_figures" if config["shapespace"]["generate_figures"] else "",
#     log:
#         "logs/shapespace_pca.log"
#     shell:
#         """
#         python -m whole_embryo_analysis.shapespace.main \
#             --feature_csv {input.feature_parquet} \
#             --outlier_path {input.outlier_csv} \
#             --save_dir {OUTPUT_DIR} \
#             --str_tag {params.str_tag} \
#             --id_name {params.id_name} \
#             --lmax {params.lmax} \
#             --num_pcs {params.num_pcs} \
#             {params.store_flag} \
#             {params.figures_flag} \
#             2>&1 | tee {log}
#         """


# # ── 3. Somite-nucleus mapping ────────────────────────────────────────────────

# rule somite_nuclei_map:
#     input:
#         feature_parquet = FEATURE_PARQUET,
#         somite_mask     = SOMITE_MASK,
#         somite_lut      = config["somite_lut"],
#     output:
#         nuclei_map = NUCLEI_MAP_CSV,
#     params:
#         output_dir = f"{OUTPUT_DIR}/somite_data",
#         sample_id  = EMBRYO_ID,
#     log:
#         "logs/somite_nuclei_map.log"
#     shell:
#         """
#         python -m whole_embryo_analysis.somite_nuclei_map \
#             --nuclear_features_path {input.feature_parquet} \
#             --somite_mask_path {input.somite_mask} \
#             --somite_lut_path {input.somite_lut} \
#             --output_path {params.output_dir} \
#             --sample_id {params.sample_id} \
#             2>&1 | tee {log}
#         """


# # ── 4. Somite feature extraction ────────────────────────────────────────────

# rule somite_features:
#     input:
#         feature_parquet = FEATURE_PARQUET,
#         outlier_csv     = OUTLIER_CSV,
#         shapespace_csv  = SHAPESPACE_CSV,
#         nuclei_map      = NUCLEI_MAP_CSV,
#         somite_mask     = SOMITE_MASK,
#         somite_lut      = config["somite_lut"],
#         nuclear_mask    = NUCLEAR_MASK,
#         bbox_parquet    = BBOX_PARQUET,
#     output:
#         positions_csv = POSITIONS_CSV,
#         somite_csv    = SOMITE_CSV,
#     params:
#         save_dir          = SOMITE_DIR,
#         embryo_id         = config["embryo_id"],
#         frame_type        = "hybrid",
#         spline_csv_path   = SPLINE_CSV,
#         positions_csv_name = config["somite"]["positions_csv_name"],
#         foldback_k        = config["somite"].get("foldback_k", 500),
#         foldback_n_iter   = config["somite"].get("foldback_n_iter", 10),
#         voxel_size        = config["somite"].get("voxel_size", [1.0, 1.0, 1.0]),
#         camera_angles     = config["somite"].get("camera_angles", []),
#         camera_center     = config["somite"].get("camera_center", []),
#         camera_zoom       = config["somite"].get("camera_zoom", ""),
#         camera_s_scale    = config["somite"].get("camera_s_scale", 4.0),
#         dask              = config["somite"]["dask"],
#     log:
#         "logs/somite_features.log"
#     shell:
#         """
#         python -m whole_embryo_analysis.somite_analysis.feature_extract \
#             --feature_path {input.feature_parquet} \
#             --outlier_path {input.outlier_csv} \
#             --nuclei_map_path {input.nuclei_map} \
#             --pc_metadata_path {input.shapespace_csv} \
#             --somite_mask_path {input.somite_mask} \
#             --somite_lut_path {input.somite_lut} \
#             --nuclear_mask_path {input.nuclear_mask} \
#             --bbox_path {input.bbox_parquet} \
#             --save_dir {params.save_dir} \
#             --embryo_id {params.embryo_id} \
#             --frame_type {params.frame_type} \
#             --spline_csv_path {params.spline_csv_path} \
#             --positions_csv_name {params.positions_csv_name} \
#             --foldback_k {params.foldback_k} \
#             --foldback_n_iter {params.foldback_n_iter} \
#             --voxel_size {params.voxel_size[0]} {params.voxel_size[1]} {params.voxel_size[2]} \
#             --camera_angles {params.camera_angles[0]} {params.camera_angles[1]} {params.camera_angles[2]} \
#             --camera_center {params.camera_center[0]} {params.camera_center[1]} {params.camera_center[2]} \
#             --camera_zoom {params.camera_zoom} \
#             --camera_s_scale {params.camera_s_scale} \
#             --dask_workers {params.dask[num_workers]} \
#             --dask_memory {params.dask[memory]} \
#             --dask_cores {params.dask[cores]} \
#             --dask_processes {params.dask[processes]} \
#             --dask_project {params.dask[project]} \
#             --dask_queue {params.dask[queue]} \
#             --dask_runtime {params.dask[runtime]} \
#             --dask_resource_spec {params.dask[resource_spec]} \
#             --dask_conda_env {params.dask[conda_env]} \
#             --dask_dashboard_port {params.dask[dashboard_port]} \
#             --dask_log_dir {params.dask[log_dir]} \
#             2>&1 | tee {log}
#         """


# # ── 5. Somite clustering ────────────────────────────────────────────────────

# rule somite_clustering:
#     input:
#         somite_csv   = SOMITE_CSV,
#         nuclear_mask = NUCLEAR_MASK,
#         somite_mask  = SOMITE_MASK,
#     output:
#         cluster_csv = CLUSTER_CSV,
#     params:
#         save_dir    = SOMITE_DIR,
#         embryo_id   = config["embryo_id"],
#         n_clusters  = config["clustering"]["n_clusters"],
#         mip_stage   = config["clustering"].get("mip_stage", 13),
#     log:
#         "logs/somite_clustering.log"
#     shell:
#         """
#         python -m whole_embryo_analysis.somite_analysis.somite_cluster \
#             --somite_csv {input.somite_csv} \
#             --save_dir {params.save_dir} \
#             --embryo_id {params.embryo_id} \
#             --n_clusters {params.n_clusters} \
#             --nuclear_mask_path {input.nuclear_mask} \
#             --somite_mask_path {input.somite_mask} \
#             --mip_stage {params.mip_stage} \
#             2>&1 | tee {log}
#         """


# # ── 6. Compress surfaces (Dask) ─────────────────────────────────────────────

# if HAS_VIZ:
#     rule compress_surfaces:
#         input:
#             segmentation_mask = SEG_MASK,
#             somite_csv        = SOMITE_CSV,
#         output:
#             pvd_file = PVD_FILE,
#         params:
#             surface_dir       = SURFACE_DIR,
#             downsample        = config["visualization"]["downsample"],
#             decimate          = config["visualization"]["decimate"],
#             mask_subpath_flag = (
#                 f"--mask_subpath {config['visualization']['mask_subpath']}"
#                 if config["visualization"].get("mask_subpath") else ""
#             ),
#             dask              = config["visualization"]["dask"],
#         log:
#             "logs/compress_surfaces.log"
#         shell:
#             """
#             python -m whole_embryo_analysis.mask_visualization.compress_surfaces \
#                 --zarr_path {input.segmentation_mask} \
#                 --output_dir {params.surface_dir} \
#                 --downsample {params.downsample} \
#                 --decimate {params.decimate} \
#                 {params.mask_subpath_flag} \
#                 --dask_workers {params.dask[num_workers]} \
#                 --dask_memory {params.dask[memory]} \
#                 --dask_cores {params.dask[cores]} \
#                 --dask_processes {params.dask[processes]} \
#                 --dask_project {params.dask[project]} \
#                 --dask_queue {params.dask[queue]} \
#                 --dask_runtime {params.dask[runtime]} \
#                 --dask_resource_spec {params.dask[resource_spec]} \
#                 --dask_conda_env {params.dask[conda_env]} \
#                 --dask_dashboard_port {params.dask[dashboard_port]} \
#                 --dask_log_dir {params.dask[log_dir]} \
#                 2>&1 | tee {log}
#             """


#     # ── 7. Render segmentations (ParaView pvbatch + MPI) ─────────────────────

#     rule render_segmentations:
#         input:
#             pvd_file = PVD_FILE,
#         output:
#             done = touch(RENDER_DONE),
#         params:
#             paraview_bin = config["visualization"]["paraview_bin_dir"],
#             surface_dir  = SURFACE_DIR,
#             render_dir   = RENDER_DIR,
#             embryo_id    = config["embryo_id"],
#             num_frames   = config["visualization"]["num_frames"],
#             output_tag   = config["visualization"]["output_tag"],
#             mpi_slots    = config["visualization"]["mpi_slots"],
#             script       = srcdir("src/whole_embryo_analysis/mask_visualization/render_segmentations.py"),
#         log:
#             "logs/render_segmentations.log"
#         shell:
#             """
#             cd {params.paraview_bin} && \
#             mpirun -np {params.mpi_slots} pvbatch {params.script} \
#                 --embryonic_number {params.embryo_id} \
#                 --output_dir {params.render_dir} \
#                 --surface_dir {params.surface_dir} \
#                 --num_frames {params.num_frames} \
#                 --output_tag {params.output_tag} \
#                 2>&1 | tee {log}
#             """
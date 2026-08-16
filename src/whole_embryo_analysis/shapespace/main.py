import argparse
import logging

from whole_embryo_analysis.shapespace.shapespace import ShapeSpace
from whole_embryo_analysis.utils.io import scan_features
import polars as pl

"""
Driver script for shapespace analysis.
Loads SH coefficient data, optionally filters outliers, then runs PCA-based
ShapeSpace workflow (mesh reconstruction, figures, CSV export).
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_shcoeffs(full_path, outlier_path, id_name):
    """
    Load spherical harmonic coefficients from CSV with optional outlier filtering.

    Args:
        full_path: Path to full dataset CSV
        outlier_path: Path to outliers CSV (or None to skip filtering)
        id_name: Column name used as the sample identifier

    Returns:
        DataFrame of shcoeffs indexed by id_name
    """
    logger.info("Loading data...")
    query = scan_features(full_path)

    if outlier_path is not None:
        outlier_ids = pl.read_csv(outlier_path)[id_name].to_list()
        query = query.filter(~pl.col(id_name).is_in(outlier_ids))

    schema = query.collect_schema()
    print(schema)
    shcoeff_cols = [c for c in schema.names() if 'shcoeffs_' in c]
    needed_cols = [id_name] + shcoeff_cols

    df = query.select(needed_cols).collect()
    logger.info(f"Loaded {len(df)} samples")

    label_index = df[id_name].to_pandas()
    return df.select(shcoeff_cols).to_pandas().set_index(label_index)


def run_shapespace(shcoeffs_df, save_dir, str_tag, id_name, generate_figures=True, **kwargs):
    """
    Create ShapeSpace instance and run workflow.

    Args:
        shcoeffs_df: DataFrame of spherical harmonic coefficients
        save_dir: Directory to save results
        str_tag: Tag for this analysis run
        id_name: Column name for sample identifiers
        generate_figures: Whether to generate visualizations
        **kwargs: Additional arguments for ShapeSpace
    """
    shapespace = ShapeSpace(
        shcoeffs_df=shcoeffs_df,
        save_dir=save_dir,
        generate_figures=generate_figures,
        str_tag=str_tag,
        id_name=id_name,
        **kwargs
    )
    shapespace.workflow(shcoeffs_df)
    return shapespace


def run_global(full_path, outlier_path, save_dir, str_tag, id_name,
               store_results=False, generate_figures=False, **kwargs):
    logger.info(f"\n{'='*50}")
    logger.info("Processing global dataset")
    logger.info(f"{'='*50}")
    shcoeffs_df = load_shcoeffs(
        full_path=full_path,
        outlier_path=outlier_path,
        id_name=id_name,
    )
    logger.info(f"Loaded {len(shcoeffs_df)} samples")
    shapespace = run_shapespace(
        shcoeffs_df=shcoeffs_df,
        save_dir=save_dir,
        str_tag=str_tag,
        id_name=id_name,
        store_results=store_results,
        generate_figures=generate_figures,
        **kwargs
    )
    return shapespace


def parse_args():
    parser = argparse.ArgumentParser(
        description="PCA on spherical harmonic coefficients (shapespace)"
    )
    parser.add_argument("--feature_csv", required=True,
                        help="Path to full dataset CSV with shcoeffs_* columns")
    parser.add_argument("--outlier_path", default=None,
                        help="Path to outlier CSV (rows to exclude). Omit to use all data.")
    parser.add_argument("--save_dir", required=True,
                        help="Root directory for output (Shapemodes_<tag>/ created inside)")
    parser.add_argument("--str_tag", required=True,
                        help="Tag identifying this analysis run (used in folder/file names)")
    parser.add_argument("--id_name", default="Nuclear_ID",
                        help="Column name for sample identifiers")
    parser.add_argument("--lmax", type=int, default=16,
                        help="Maximum degree for spherical harmonic reconstruction")
    parser.add_argument("--num_pcs", type=int, default=6,
                        help="Number of principal components to compute")
    parser.add_argument("--store_results", action="store_true",
                        help="Write shapespace CSV and VTK meshes to disk")
    parser.add_argument("--generate_figures", action="store_true",
                        help="Generate PyVista / matplotlib visualizations")
    return parser.parse_args()


def main():
    args = parse_args()
    run_global(
        full_path=args.feature_csv,
        outlier_path=args.outlier_path,
        save_dir=args.save_dir,
        str_tag=args.str_tag,
        id_name=args.id_name,
        store_results=args.store_results,
        generate_figures=args.generate_figures,
        lmax=args.lmax,
        num_pcs=args.num_pcs,
    )


if __name__ == "__main__":
    main()

"""
python3 main.py --feature_csv /net/beliveau/vol2/instrument/E9.5_306/Zoom_306/dataset_fused_features-JGseg-030926.csv --outlier_path /net/beliveau/vol2/instrument/E9.5_306/Zoom_306/analysis_pipeline/KDE/outliers.csv --save_dir /net/beliveau/vol1/home/msforman/msf_project/whole_embryo_analysis2/logs/ --str_tag E9.5_306 --id_name Nuclear_ID --lmax 16 --num_pcs 6 --store_results --generate_figures
python3 main.py --feature_csv /net/beliveau/vol2/instrument/E9.5_306/Zoom_306/dataset_fused_features_Julia_FT_Feb26.csv --outlier_path /net/beliveau/vol2/instrument/E9.5_306/Zoom_306/analysis_pipeline/KDE/outliers.csv --save_dir /net/beliveau/vol1/home/msforman/msf_project/whole_embryo_analysis2/logs/ --str_tag E9.5_306_Julia_FT --id_name Nuclear_ID --lmax 16 --num_pcs 6 --store_results --generate_figures
"""
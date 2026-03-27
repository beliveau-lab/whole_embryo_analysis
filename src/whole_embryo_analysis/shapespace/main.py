from shapespace import ShapeSpace
import polars as pl

"""
This is the driver script for shapespace analysis. It will load the data, create the ShapeSpace object, and run ShapeSpace.workflow().
"""

def load_shcoeffs(full_path, outlier_path, id_name):
    """
    Load spherical harmonic coefficients from CSV with optional filtering of outliers.
    if you don't want to filter outliers, set outlier_path to None.
    
    Args:
        full_path: Path to full dataset CSV
        outlier_path: Path to outliers CSV (or None)
        class_column: Column name to filter by (optional)
        class_value: Value to filter for (optional)
    
    Returns:
        DataFrame of shcoeffs indexed by Nuclear_ID
    """
    print("Loading data...")
    query = pl.scan_csv(full_path)

    if outlier_path is not None:
        outlier_ids = pl.read_csv(outlier_path)[id_name].to_list()
        query = query.filter(~pl.col(id_name).is_in(outlier_ids))

    schema = query.collect_schema()
    shcoeff_cols = [c for c in schema.names() if 'shcoeffs_' in c]
    needed_cols = [id_name] + shcoeff_cols
    
    df = query.select(needed_cols).collect()
    print(f"Loaded {len(df)} samples")
    
    label_index = df[id_name].to_pandas()
    return df.select(shcoeff_cols).to_pandas().set_index(label_index)


def run_shapespace(shcoeffs_df, save_dir, str_tag, id_name, generate_figures=True, **kwargs):
    """
    Create ShapeSpace instance and run workflow.
    
    Args:
        shcoeffs_df: DataFrame of spherical harmonic coefficients
        save_dir: Directory to save results
        str_tag: Tag for this analysis
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

def run_global(full_path, outlier_path, save_dir, str_tag, id_name, store_results=False, generate_figures=False):
    print(f"\n{'='*50}")
    print("Processing global dataset")
    print(f"{'='*50}")
    shcoeffs_df = load_shcoeffs(
        full_path=full_path,
        outlier_path=outlier_path,
        id_name=id_name,
    )
    print(f"Loaded {len(shcoeffs_df)} samples")
    shapespace = run_shapespace(
        shcoeffs_df=shcoeffs_df,
        save_dir=save_dir,
        str_tag=str_tag,
        id_name=id_name,
        store_results=store_results,
        generate_figures=generate_figures
    )
    return shapespace

def main():
    # full_path = '/net/beliveau/vol2/instrument/E9.5_306/Zoom_306/dataset_fused_features_postprocessed.csv'
    # full_path = '/net/beliveau/vol2/instrument/E9.5_306/Zoom_306/dataset_fused_features_postprocessed_newclasslabels.csv'
    # outlier_path = '/net/beliveau/vol2/instrument/E9.5_306/Zoom_306/outliers_postprocessed.csv'
    full_path = '/net/beliveau/vol2/instrument/E9.5_306/Zoom_306/dataset_fused_features_Julia_FT_Feb26.csv'
    outlier_path = None
    save_dir = '/net/beliveau/vol2/instrument/E9.5_306/Zoom_306/'
    (
    run_global(
        full_path, outlier_path, save_dir, 
        id_name='Nuclear_ID', str_tag='Julia_FT_Feb26', store_results=True, 
        generate_figures=True
        )
    )

if __name__ == "__main__":
    main()


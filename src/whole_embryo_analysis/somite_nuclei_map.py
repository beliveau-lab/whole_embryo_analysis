"""
Assign each nucleus a Somite_ID by looking up its (scaled) centroid
position in a somite label mask.

Both the mask and the centroids are in s3 (8x-downsampled) space so
no full-resolution arrays are ever loaded.
"""
import tifffile
import numpy as np
import polars as pl
import argparse
from whole_embryo_analysis.utils.scale_resolution import S3_SCALE, from_s0, scale_mask
from whole_embryo_analysis.somite_analysis.bio_frame import get_somite_centroids
import os

def load_nuclear_features(csv_path, use_s0=False): # if use_s0 False -> s3
    """Load centroids and Nuclear_ID from a feature CSV."""
    coord_cols = ["Centroid_X", "Centroid_Y", "Centroid_Z"]
    df = (
        pl.scan_csv(csv_path)
        .select(coord_cols + ["Nuclear_ID"])
        .collect()
    )
    if not use_s0:
        scaled = df.clone()
        for c in coord_cols:
            arr = from_s0(df[c].to_numpy(), S3_SCALE, as_index=True, rounding="round")
            scaled = scaled.with_columns(pl.Series(c, arr))
        return scaled
    else:
        return df

def assign_somite_ids(df, somite_mask):
    """Add a ``Somite_ID`` column by indexing into *somite_mask* at each centroid.

    Parameters
    ----------
    df : pl.DataFrame
        Nuclear data frame already scaled such that centroids match scale of 
        masks
    somite_mask : np.ndarray, shape (Z, Y, X)
        Label image where each somite has a unique positive integer and
        background is 0.
    Returns
    -------
    pl.DataFrame
        Copy of *df* with a new ``Somite_ID`` column (UInt32).
        Nuclei outside every somite get ``null``.
    """
    z_nucleus = df["Centroid_Z"].to_numpy()
    y_nucleus = df["Centroid_Y"].to_numpy()
    x_nucleus = df["Centroid_X"].to_numpy()

    mz, my, mx = somite_mask.shape

    # Clamp to valid mask indices so we don't go out of bounds.
    z_safe = np.clip(z_nucleus, 0, mz - 1)
    y_safe = np.clip(y_nucleus, 0, my - 1)
    x_safe = np.clip(x_nucleus, 0, mx - 1)
    # Vectorised look-up: one array index per nucleus.
    ids = somite_mask[z_safe, y_safe, x_safe]
    # Also flag nuclei that were out of bounds as background (0).
    out_of_bounds = (
        (z_nucleus < 0) | (z_nucleus >= mz) |
        (y_nucleus < 0) | (y_nucleus >= my) |
        (x_nucleus < 0) | (x_nucleus >= mx)
    )    
    ids[out_of_bounds] = 0
    # 0 → null, positive → somite ID
    somite_col = pl.Series("Somite_ID", ids, dtype=pl.UInt32)
    somite_col = somite_col.replace(0, None)
    return df.with_columns(somite_col)



def validate_centroid_proximity(nuc_df, somite_lut, max_distance=50.0):
    """Check that mean nuclear centroids are close to somite mask centroids.
    Parameters
    ----------
    nuc_df : pl.DataFrame
        Nuclear data with ``Somite_ID``, ``Centroid_Z/Y/X`` (s3 scale).
    somite_lut : pl.DataFrame
        Somite look-up table with ``Somite_ID`` and ``Centroid_Z/Y/X``
        columns in s3 space (as produced by ``get_somite_centroids``).
    max_distance : float
        Euclidean distance threshold (in s3 voxels) above which a warning
        is printed.  Default 50.
    Returns
    -------
    pl.DataFrame
        Per-somite table with ``Somite_ID``, ``nuc_z/y/x``,
        ``mask_z/y/x``, and ``distance``.
    """
    # Mean nuclear centroid per somite (s3 space)
    nuc_centroids = (
        nuc_df
        .filter(pl.col("Somite_ID").is_not_null())
        .group_by("Somite_ID")
        .agg(
            pl.col("Centroid_Z").mean().alias("nuc_z"),
            pl.col("Centroid_Y").mean().alias("nuc_y"),
            pl.col("Centroid_X").mean().alias("nuc_x"),
        )
    )

    mask_centroids = somite_lut.select(
        pl.col("Somite_ID").cast(pl.UInt32),
        pl.col("Centroid_Z").alias("mask_z"),
        pl.col("Centroid_Y").alias("mask_y"),
        pl.col("Centroid_X").alias("mask_x"),
    )

    comparison = nuc_centroids.join(mask_centroids, on="Somite_ID", how="inner")
    comparison = comparison.with_columns(
        (
            (pl.col("nuc_z") - pl.col("mask_z")).pow(2)
            + (pl.col("nuc_y") - pl.col("mask_y")).pow(2)
            + (pl.col("nuc_x") - pl.col("mask_x")).pow(2)
        ).sqrt().alias("distance")
    )

    failures = comparison.filter(pl.col("distance") > max_distance)
    if len(failures) > 0:
        print(f"WARNING: {len(failures)} somite(s) exceed {max_distance}-voxel "
              f"centroid distance threshold:")
        print(failures.select("Somite_ID", "distance"))
    else:
        print(f"Centroid proximity check passed (all {len(comparison)} somites "
              f"within {max_distance} voxels).")

    return comparison


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nuclear_features_path", type=str, required=True)
    parser.add_argument("--somite_mask_path", type=str, required=True)
    parser.add_argument("--somite_lut_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--sample_id", type=str, required=True)
    parser.add_argument("--use_s0", action="store_true", help="Use s0 coordinates for assigning nuclei to somites; if omitted, s3 coordinates are used. This will take far more memory and time.")
    return parser.parse_args()

def check_lut_ids(somite_lut, somite_mask):
    ids_mask = np.unique(somite_mask).astype(int)
    ids_mask = [id for id in ids_mask if id != 0]
    ids_lut = somite_lut['Somite_ID'].unique().to_numpy()
    ids_mask = set(ids_mask)
    ids_lut = set(ids_lut)
    set_diff = ids_mask - ids_lut
    if len(set_diff) != 0:
        raise ValueError(f"Unique ids in mask and lut do not match (mask_ids - lut_ids): {set_diff}")
        return False
    else:
        print("Sanity check passed, Somite IDs in LUT and mask match.")
        return True

def check_nuclear_somite_ids(nuc_df, somite_lut):
    nuclear_somite_ids = nuc_df["Somite_ID"].unique().to_numpy()
    lut_somite_ids = somite_lut["Somite_ID"].unique().to_numpy()
    set_diff = set(lut_somite_ids) - set(nuclear_somite_ids)
    if len(set_diff) != 0:
        raise ValueError(f"Unique ids in lut and nuclear df do not match (lut_ids - nuclear_ids): {set_diff}")
        return False
    else:
        print("Sanity check passed, Somite IDs in LUT and nuclear df match.")
        return True

def main():
    args = parse_args()
    # load nuclear features and scale if needed
    nuc_df = load_nuclear_features(args.nuclear_features_path, use_s0=args.use_s0)
    total_nuclei = len(nuc_df)
    # load somite mask and LUT
    somite_mask = tifffile.imread(args.somite_mask_path)
    if args.use_s0:
        print("Scaling somite mask to s3 resolution...")
        somite_mask = scale_mask(somite_mask, S3_SCALE)
    
    somite_lut = pl.read_csv(args.somite_lut_path)
    somite_lut = somite_lut.filter(pl.col("Embryo_ID") == int(args.sample_id))
    print(f"Number of total somites in {args.sample_id}: {len(somite_lut["Somite_ID"].unique())}")
    # Sanity check on ids in each
    check_lut_ids(somite_lut, somite_mask)
    somite_lut = get_somite_centroids(somite_mask, somite_lut, scale=False)
    # Assign somite IDs to nuclei
    nuc_df = assign_somite_ids(nuc_df, somite_mask)
    somite_nuclei = len(nuc_df.filter(pl.col("Somite_ID").is_not_null()))
    # Sanity check on ids in nuclear df
    check_nuclear_somite_ids(nuc_df, somite_lut)
    # Validate centroid proximity
    validate_centroid_proximity(nuc_df, somite_lut)
    print(f"Number of nuclei with somite IDs: {somite_nuclei}")
    print(f"Number of nuclei with no somite IDs: {total_nuclei - somite_nuclei}")
    # Save output
    if args.use_s0:
        file_name = f"{args.sample_id}_somite_nuclei_map_s0"
    else:
        file_name = f"{args.sample_id}_somite_nuclei_map"
    nuc_df = nuc_df.drop(["Centroid_Z", "Centroid_Y", "Centroid_X"])
    nuc_df.write_csv(f'{args.output_path}/{file_name}.csv')

if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""Stream one or more CSVs to Parquet.

Extracted from the features_to_parquet Snakemake rule so that failures produce
a readable traceback instead of an empty log. Pairs are positional so the rule
stays a one-liner:

    python scripts/csv_to_parquet.py \
        --pair features.csv features.parquet \
        --pair bboxes.csv bboxes.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--pair", nargs=2, metavar=("SRC", "DST"), action="append", required=True,
        help="CSV source and Parquet destination. Repeatable.",
    )
    p.add_argument(
        "--infer-schema-length", type=int, default=10000,
        help="Rows scanned to infer column dtypes (default: %(default)s). "
             "Use 0 to scan the whole file.",
    )
    p.add_argument(
        "--compression", default="zstd",
        choices=["zstd", "snappy", "gzip", "lz4", "brotli", "uncompressed"],
        help="Parquet codec (default: %(default)s).",
    )
    p.add_argument(
        "--no-fallback", action="store_true",
        help="Fail immediately if streaming sink fails, instead of retrying "
             "with an in-memory collect.",
    )
    return p.parse_args(argv)


def convert(src: Path, dst: Path, *, infer_schema_length: int,
            compression: str, allow_fallback: bool) -> None:
    """Convert one CSV to Parquet, streaming if possible."""
    if not src.is_file():
        raise FileNotFoundError(f"input CSV does not exist: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)

    size_gb = src.stat().st_size / 1e9
    print(f"Converting {src} -> {dst}  ({size_gb:.2f} GB)", flush=True)

    lf = pl.scan_csv(
        src,
        infer_schema_length=infer_schema_length or None,
    )

    try:
        lf.sink_parquet(dst, compression=compression)
    except Exception as exc:
        if not allow_fallback:
            raise
        # Streaming sink can fail on schema surprises or unsupported plans.
        # The fallback materializes the frame, so it needs RAM >= table size.
        print(f"sink_parquet failed ({type(exc).__name__}: {exc})", flush=True)
        print(f"falling back to in-memory collect (needs ~{size_gb:.1f} GB+ RAM)",
              flush=True)
        _collect_fallback(lf).write_parquet(dst, compression=compression)

    out_gb = dst.stat().st_size / 1e9
    print(f"Wrote {dst}  ({out_gb:.2f} GB)", flush=True)


def _collect_fallback(lf: pl.LazyFrame) -> pl.DataFrame:
    """collect() across the polars API change.

    The `streaming=True` keyword was deprecated then removed in polars 1.x in
    favour of `engine="streaming"`. Try the new form first, then the old, then
    plain collect.
    """
    for kwargs in ({"engine": "streaming"}, {"streaming": True}, {}):
        try:
            return lf.collect(**kwargs)
        except TypeError:
            continue
    raise RuntimeError("no working collect() signature found")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print(f"polars {pl.__version__}", flush=True)

    for src, dst in args.pair:
        convert(
            Path(src), Path(dst),
            infer_schema_length=args.infer_schema_length,
            compression=args.compression,
            allow_fallback=not args.no_fallback,
        )

    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
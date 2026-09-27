"""Reading the challenge TSVs and writing submission files."""
from __future__ import annotations

import os
from pathlib import Path

import polars as pl

COLS = ["entity_id", "business_name", "business_address", "country"]


def read_source(path: str | Path) -> pl.DataFrame:
    """Read one *_sourceN.tsv file. Quotes are not special in these files."""
    df = pl.read_csv(
        path,
        separator="\t",
        quote_char=None,
        schema_overrides={c: pl.Utf8 for c in COLS},
    )
    return df.with_columns(pl.col(c).fill_null("") for c in COLS[1:])


def read_split(data_dir: str | Path, split: str, cache_dir: str | Path | None = None) -> dict[int, pl.DataFrame]:
    """Return {1: s1, 2: s2, 3: s3} for split in {"train", "test"}, caching as parquet."""
    out = {}
    for s in (1, 2, 3):
        cache = Path(cache_dir) / f"{split}_s{s}.parquet" if cache_dir else None
        if cache is not None and cache.exists():
            out[s] = pl.read_parquet(cache)
            continue
        df = read_source(Path(data_dir) / split / f"{split}_source{s}.tsv")
        if cache is not None:
            os.makedirs(cache.parent, exist_ok=True)
            df.write_parquet(cache)
        out[s] = df
    return out


def read_ground_truth(path: str | Path) -> pl.DataFrame:
    """Return long (s1_id, match_id) pairs from train_ground_truth.tsv."""
    gt = pl.read_csv(
        path,
        separator="\t",
        quote_char=None,
        schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8},
    ).with_columns(pl.col("matched_entity_ids").fill_null(""))
    return (
        gt.select(
            pl.col("source1_entity_id").alias("s1_id"),
            pl.col("matched_entity_ids").str.split(",").alias("match_id"),
        )
        .explode("match_id")
        .filter(pl.col("match_id") != "")
    )


def write_id_lists(s1_ids: pl.Series, pairs: pl.DataFrame, value_col: str, path: str | Path) -> None:
    """Write `source1_entity_id<TAB>value_col` with one row per S1 id (empty list allowed).

    `pairs` has columns s1_id, rid (S2/S3 entity id).
    """
    grouped = (
        pairs.select("s1_id", "rid")
        .unique()
        .sort(["s1_id", "rid"])
        .group_by("s1_id", maintain_order=True)
        .agg(pl.col("rid").str.join(",").alias(value_col))
    )
    out = (
        pl.DataFrame({"source1_entity_id": s1_ids})
        .join(grouped, left_on="source1_entity_id", right_on="s1_id", how="left")
        .with_columns(pl.col(value_col).fill_null(""))
    )
    os.makedirs(Path(path).parent, exist_ok=True)
    # Hand-rolled writer: polars would quote nothing here anyway, but keep it explicit.
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{value_col}\n")
        for a, b in out.iter_rows():
            f.write(f"{a}\t{b}\n")

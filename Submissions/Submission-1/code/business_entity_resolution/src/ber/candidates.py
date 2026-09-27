"""Blocking + pair-feature extraction for a whole split, written as parquet parts.

Pass 1 (per country, per query chunk): retrieve top-K S1 candidates for every S2/S3
record, prune them, compute pair features, write `parts/<country>_<i>.parquet`.
Pass 2: add S1-side context features (competition among queries for the same S1).
"""
from __future__ import annotations

import gc
import shutil
import time
from pathlib import Path

import numpy as np
import polars as pl

from .blocking import W_COMBINED, S1Index
from .pairfeat import pair_features, record_meta

K_RETRIEVE = 20     # neighbours retrieved per query (context features use all of them)
K_KEEP = 10         # candidates kept per query ...
KEEP_RATIO = 0.6    # ... if their blocking score >= KEEP_RATIO * the query's best score
CHUNK = 1_000_000


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _name_freq(s1c: pl.DataFrame, other: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    counts = s1c.group_by("nm_core").agg(pl.len().alias("c"))
    a = s1c.select("nm_core").join(counts, on="nm_core", how="left")["c"].to_numpy()
    b = other.select("nm_core").join(counts, on="nm_core", how="left")["c"].fill_null(0).to_numpy()
    return a, b


def build(norm_paths: dict[int, Path], out_dir: str | Path, n_threads: int = 14,
          only: list[str] | None = None) -> Path:
    """norm_paths: normalised parquet files {1: s1, 2: s2, 3: s3}. Returns the parts directory.

    Countries are taken from the data (open set); each is processed independently.
    `only` restricts the run to some countries (their old parts are replaced, others kept).
    """
    out_dir = Path(out_dir)
    parts_dir = out_dir / "parts"
    if only is None and parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    for c in only or []:
        for p in parts_dir.glob(f"{c}_*.parquet"):
            p.unlink()
    ctx = out_dir / "s1ctx.parquet"   # derived from all parts; stale once parts change
    if ctx.exists():
        ctx.unlink()
    s1_lf = pl.scan_parquet(norm_paths[1])
    q_lf = pl.concat([pl.scan_parquet(norm_paths[2]).with_columns(pl.lit(2, pl.Int8).alias("src")),
                      pl.scan_parquet(norm_paths[3]).with_columns(pl.lit(3, pl.Int8).alias("src"))])
    countries = sorted(
        set(s1_lf.select("country").unique().collect()["country"].to_list())
        | set(q_lf.select("country").unique().collect()["country"].to_list())
    )
    for country in countries:
        if only is not None and country not in only:
            continue
        s1c = s1_lf.filter(pl.col("country") == country).collect()
        qc = q_lf.filter(pl.col("country") == country).collect()
        if len(s1c) == 0 or len(qc) == 0:
            _log(f"{country}: skipped (s1={len(s1c)}, queries={len(qc)})")
            continue
        _log(f"{country}: indexing {len(s1c):,} S1 records, {len(qc):,} queries")
        idx = S1Index(s1c)
        s_meta = record_meta(s1c)
        s_ids = s1c["entity_id"]
        s_freq, _ = _name_freq(s1c, s1c.head(0))
        for ci, start in enumerate(range(0, len(qc), CHUNK)):
            qch = qc.slice(start, CHUNK)
            qw, qcsr = idx.encode(qch)
            res = idx.topk(qw, len(qch), K_RETRIEVE, W_COMBINED, n_threads=n_threads)
            del qw
            ctx = res.group_by("q_row").agg(
                pl.col("bscore").max().alias("top1"),
                pl.col("bscore").sort(descending=True).get(1, null_on_oob=True).fill_null(0).alias("top2"),
                pl.len().cast(pl.Int16).alias("n_ret"),
            )
            res = res.join(ctx, on="q_row").filter(
                (pl.col("brank") < K_KEEP) & (pl.col("bscore") >= KEEP_RATIO * pl.col("top1"))
            ).sort(["q_row", "brank"])
            _, q_freq = _name_freq(s1c, qch)
            qi = res["q_row"].to_numpy()
            si = res["s1_row"].to_numpy()
            feats = pair_features(qi, si, record_meta(qch), s_meta, qcsr, idx.csr, q_freq, s_freq)
            out = pl.concat([
                pl.DataFrame({
                    "q_id": qch["entity_id"].gather(qi),
                    "s1_id": s_ids.gather(si),
                    "src": qch["src"].gather(qi),
                }),
                res.select(
                    "bscore", "brank", "top1", "top2", "n_ret",
                    (pl.col("bscore") - pl.col("top1")).alias("gap_top1"),
                    (pl.col("top1") - pl.col("top2")).alias("top_margin"),
                    (pl.col("bscore") / pl.col("top1")).alias("ratio_top1"),
                ),
                feats,
            ], how="horizontal")
            out.write_parquet(parts_dir / f"{country}_{ci:03d}.parquet")
            _log(f"{country} chunk {ci}: {len(qch):,} queries -> {len(out):,} pairs")
            del res, feats, out, qcsr
            gc.collect()
        del idx, s_meta, s1c, qc
        gc.collect()
    return parts_dir


def s1_context(parts_dir: str | Path) -> pl.DataFrame:
    """Per-pair S1-side features: how many queries compete for the same S1 entity and
    where this query ranks among them."""
    lf = pl.scan_parquet(str(Path(parts_dir) / "*.parquet")).select("q_id", "s1_id", "bscore", "brank")
    return lf.with_columns(
        pl.len().over("s1_id").cast(pl.Int32).alias("s1_nq"),
        (pl.col("brank") == 0).sum().over("s1_id").cast(pl.Int32).alias("s1_ntop1"),
        pl.col("bscore").max().over("s1_id").alias("s1_best"),
        pl.col("bscore").rank("ordinal", descending=True).over("s1_id").cast(pl.Int32).alias("s1_rank"),
    ).with_columns(
        (pl.col("bscore") - pl.col("s1_best")).alias("s1_gap"),
    ).select("q_id", "s1_id", "s1_nq", "s1_ntop1", "s1_rank", "s1_gap").collect()

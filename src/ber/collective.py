"""Collective "shared deviation" features: is a record's difference from an S1 entity
idiosyncratic noise, or the signature of a separate entity that other records share?

Copy noise is applied independently per record, so the name tokens by which a true copy
deviates from its S1 (typos, junk words) are rarely repeated by the entity's other
records. A distractor or orphan *entity* (a different business, absent from S1, whose
records land on a similar S1) has several records that all carry the same deviation:
the word that differs, or the other house number. Pairwise features cannot see this; it
is a property of all records claiming the same S1.

For a pair (q, s), over all records r != q that have s among their candidates:
  cd_share         max over q's name tokens absent from s of #r containing that token
  cd_share_strong  the same, counting only r with stage-1 p1(r, s) >= 0.5
  cd_ndev          number of q's name tokens absent from s
  cd_num_diff      q's first house number differs from s's (both present)
  cd_num_share     #r carrying the same differing house number
Ids and tokens are joined as 64-bit hashes to keep memory low.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

COLL = ("cd_share", "cd_share_strong", "cd_ndev", "cd_num_diff", "cd_num_share")
_SEED = 97


def record_tokens(work: Path, split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    norm = work / "norm"
    cols = ["entity_id", "nm_tok", "ad_first_num"]
    q = pl.concat([pl.read_parquet(norm / f"{split}_s{s}.parquet", columns=cols) for s in (2, 3)])
    s1 = pl.read_parquet(norm / f"{split}_s1.parquet", columns=cols)
    return q, s1


def _tok(df: pl.DataFrame, key: str) -> pl.DataFrame:
    return (df.select(pl.col("entity_id").hash(_SEED).alias(key), pl.col("nm_tok").alias("t")).explode("t")
            .drop_nulls().select(key, pl.col("t").hash(_SEED).alias("th")).unique())


def collective_features(pairs: pl.DataFrame, q: pl.DataFrame, s1: pl.DataFrame) -> pl.DataFrame:
    """pairs: (q_id, s1_id, p1) of one split / world -> (q_id, s1_id, *COLL)."""
    P = pairs.select("q_id", "s1_id", pl.col("q_id").hash(_SEED).alias("qh"), pl.col("s1_id").hash(_SEED).alias("sh"),
                     (pl.col("p1") >= 0.5).cast(pl.UInt32).alias("st"))
    dev = (P.select("qh", "sh", "st").join(_tok(q, "qh"), on="qh")
           .join(_tok(s1, "sh"), on=["sh", "th"], how="anti"))
    cnt = dev.group_by("sh", "th").agg(pl.len().cast(pl.UInt32).alias("n"), pl.col("st").sum().alias("ns"))
    agg = (dev.join(cnt, on=["sh", "th"])
           .group_by("qh", "sh").agg((pl.col("n") - 1).max().alias("cd_share"),
                                     (pl.col("ns") - pl.col("st")).max().alias("cd_share_strong"),
                                     pl.len().alias("cd_ndev")))
    del dev, cnt
    qn = q.select(pl.col("entity_id").hash(_SEED).alias("qh"), pl.col("ad_first_num").hash(_SEED).alias("qn"),
                  pl.col("ad_first_num").is_not_null().alias("qok"))
    sn = s1.select(pl.col("entity_id").hash(_SEED).alias("sh"), pl.col("ad_first_num").hash(_SEED).alias("sn"),
                   pl.col("ad_first_num").is_not_null().alias("sok"))
    num = (P.select("qh", "sh").join(qn, on="qh", how="left").join(sn, on="sh", how="left")
           .with_columns((pl.col("qok").fill_null(False) & pl.col("sok").fill_null(False)
                          & (pl.col("qn") != pl.col("sn"))).alias("nd")))
    nc = num.filter("nd").group_by("sh", "qn").agg(pl.len().cast(pl.Int64).alias("nn"))
    num = num.join(nc, on=["sh", "qn"], how="left").select(
        "qh", "sh", pl.col("nd").cast(pl.Float32).alias("cd_num_diff"),
        pl.when(pl.col("nd")).then(pl.col("nn") - 1).otherwise(0).cast(pl.Float32).alias("cd_num_share"))
    out = (P.select("q_id", "s1_id", "qh", "sh").join(num, on=["qh", "sh"], how="left")
           .join(agg, on=["qh", "sh"], how="left")
           .with_columns([pl.col(c).fill_null(0).cast(pl.Float32) for c in ("cd_share", "cd_share_strong", "cd_ndev")])
           .select("q_id", "s1_id", *COLL))
    return out

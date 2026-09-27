"""Difference-type features for candidate pairs (run after `candidates`).

Distractors in this data are *mutations of real S1 entities*, and their mutations differ
in kind from the noise applied to genuine copies:

  name     distractor: a word replaced by another real vocabulary word (Tech -> Leather)
           true copy : character typos, dropped / shuffled / abbreviated words
  legal    distractor: switched to a different legal form (Private Ltd -> Public Ltd)
           true copy : legal form abbreviated, dropped or re-ordered
  numbers  distractor: small arithmetic change (12590 -> 12594, 102/2A -> 102/7A)
           true copy : truncation / digit typo (4712 -> 471, 2A/84 -> 2A/8)

So besides *how similar* a pair is, these features describe *what kind of difference*
separates it: unmatched name tokens are split into typo-matched ones and substitutions
(known vocabulary words from the S1 name vocabulary), one-sided legal forms are listed,
and the closest unit-code pair is characterised. Output: `work/<split>/extra/<part>`.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import process
from rapidfuzz.distance import JaroWinkler, Levenshtein

ID_COLS = ["q_id", "s1_id"]
LEGAL = ["pvt", "ltd", "llc", "llp", "inc", "corp", "co", "pc", "lp", "plc", "pllc", "public",
         "sarl", "sas", "sa", "eurl", "sci", "snc"]
TYPO_JW = 0.8        # unmatched tokens at least this similar are treated as a typo pair
KNOWN_DF = 20        # a token used by >= KNOWN_DF S1 names counts as a vocabulary word


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _records(work: Path, split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    cols = ["entity_id", "nm_tok", "nm", "ad_code"]
    norm = work / "norm"
    q = pl.concat([pl.read_parquet(norm / f"{split}_s{s}.parquet", columns=cols) for s in (2, 3)])
    s1 = pl.read_parquet(norm / f"{split}_s1.parquet", columns=cols)
    return q, s1


def _legal(nm: pl.Expr) -> pl.Expr:
    return nm.str.split(" ").list.eval(pl.element().filter(pl.element().is_in(LEGAL))).list.unique()


def _best_cross(a: pl.DataFrame, b: pl.DataFrame, scorer, agg: str) -> pl.DataFrame:
    """For every (pid, token) of `a`, the best `scorer` value against the tokens of `b`
    with the same pid. a, b: (pid, t). Returns (pid, t, best)."""
    x = a.join(b, on="pid", suffix="_b")
    if len(x) == 0:
        return a.with_columns(pl.lit(None, pl.Float32).alias("best"))
    x = x.with_columns(pl.Series("v", process.cpdist(x["t"].to_list(), x["t_b"].to_list(),
                                                    scorer=scorer, workers=-1, dtype=np.float32)))
    fn = pl.col("v").max() if agg == "max" else pl.col("v").min()
    return x.group_by("pid", "t").agg(fn.alias("best"))


def pair_diff_features(pairs: pl.DataFrame, q: pl.DataFrame, s1: pl.DataFrame, vocab: pl.DataFrame) -> pl.DataFrame:
    """pairs: (q_id, s1_id). q / s1: (entity_id, nm_tok, nm, ad_code). vocab: (t, df)."""
    d = (
        pairs.select(ID_COLS).with_row_index("pid")
        .join(q.rename({"entity_id": "q_id"}), on="q_id", how="left")
        .join(s1.rename({"entity_id": "s1_id"}), on="s1_id", how="left", suffix="_s")
        .sort("pid")
    )
    n = len(d)
    # ---- name tokens: exact matches, typo pairs, vocabulary substitutions
    d = d.with_columns(
        pl.col("nm_tok").list.set_difference(pl.col("nm_tok_s")).alias("qx"),
        pl.col("nm_tok_s").list.set_difference(pl.col("nm_tok")).alias("sx"),
    )
    qx = d.select("pid", pl.col("qx").alias("t")).explode("t", empty_as_null=True).drop_nulls("t")
    sx = d.select("pid", pl.col("sx").alias("t")).explode("t", empty_as_null=True).drop_nulls("t")
    qb = _best_cross(qx, sx, JaroWinkler.normalized_similarity, "max").with_columns(pl.col("best").fill_null(0))
    sb = _best_cross(sx, qx, JaroWinkler.normalized_similarity, "max").with_columns(pl.col("best").fill_null(0))
    feats = {}
    for name, tb in (("q", qb), ("s", sb)):
        tb = tb.join(vocab, on="t", how="left").with_columns(pl.col("df").fill_null(0))
        g = tb.group_by("pid").agg(
            pl.len().alias("n"),
            (pl.col("best") >= TYPO_JW).sum().alias("typo"),
            ((pl.col("best") < TYPO_JW) & (pl.col("df") >= KNOWN_DF)).sum().alias("known"),
            ((pl.col("best") < TYPO_JW) & (pl.col("df") < KNOWN_DF)).sum().alias("rare"),
            pl.when(pl.col("best") < TYPO_JW).then(pl.col("df")).otherwise(None).max().alias("maxdf"),
            pl.col("best").min().alias("minjw"),
        )
        full = pl.DataFrame({"pid": np.arange(n, dtype=np.uint32)}).join(g, on="pid", how="left").sort("pid")
        feats[f"nm_{name}x_n"] = full["n"].fill_null(0).to_numpy().astype(np.float32)
        feats[f"nm_{name}x_typo"] = full["typo"].fill_null(0).to_numpy().astype(np.float32)
        feats[f"nm_{name}x_known"] = full["known"].fill_null(0).to_numpy().astype(np.float32)
        feats[f"nm_{name}x_rare"] = full["rare"].fill_null(0).to_numpy().astype(np.float32)
        feats[f"nm_{name}x_logmaxdf"] = np.log1p(full["maxdf"].cast(pl.Float64).to_numpy()).astype(np.float32)
        feats[f"nm_{name}x_minjw"] = full["minjw"].to_numpy().astype(np.float32)
    # ---- legal forms present on one side only
    lg = d.select(_legal(pl.col("nm")).alias("lq"), _legal(pl.col("nm_s")).alias("ls"))
    lg = lg.with_columns(
        pl.col("lq").list.set_difference(pl.col("ls")).alias("oq"),
        pl.col("ls").list.set_difference(pl.col("lq")).alias("os"),
    )
    feats["legal_q_only_n"] = lg["oq"].list.len().to_numpy().astype(np.float32)
    feats["legal_s_only_n"] = lg["os"].list.len().to_numpy().astype(np.float32)
    feats["legal_equal"] = ((lg["oq"].list.len() == 0) & (lg["os"].list.len() == 0)
                            & (lg["lq"].list.len() > 0)).to_numpy().astype(np.float32)
    for form in ("public", "pvt", "ltd", "llc", "inc", "corp", "co", "pc"):
        feats[f"legal_q_only_{form}"] = lg["oq"].list.contains(form).to_numpy().astype(np.float32)
        feats[f"legal_s_only_{form}"] = lg["os"].list.contains(form).to_numpy().astype(np.float32)
    # ---- unit / plot codes: closest pair
    qc = d.select("pid", pl.col("ad_code").alias("t")).explode("t", empty_as_null=True).drop_nulls("t")
    sc = d.select("pid", pl.col("ad_code_s").alias("t")).explode("t", empty_as_null=True).drop_nulls("t")
    x = qc.join(sc, on="pid", suffix="_b")
    if len(x):
        x = x.with_columns(
            pl.Series("lev", process.cpdist(x["t"].to_list(), x["t_b"].to_list(), scorer=Levenshtein.distance,
                                            workers=-1, dtype=np.int32)),
            (pl.col("t").str.starts_with(pl.col("t_b")) | pl.col("t_b").str.starts_with(pl.col("t"))
             | pl.col("t").str.ends_with(pl.col("t_b")) | pl.col("t_b").str.ends_with(pl.col("t"))).alias("affix"),
        )
        g = x.group_by("pid").agg(
            pl.col("lev").min().alias("lev"),
            pl.col("affix").filter(pl.col("lev") > 0).any().alias("affix"),
            ((pl.col("lev") == 1) & (pl.col("t").str.len_chars() == pl.col("t_b").str.len_chars())).any().alias("subst1"),
        )
    else:
        g = pl.DataFrame(schema={"pid": pl.UInt32, "lev": pl.Int32, "affix": pl.Boolean, "subst1": pl.Boolean})
    full = pl.DataFrame({"pid": np.arange(n, dtype=np.uint32)}).join(g, on="pid", how="left").sort("pid")
    feats["code_minlev"] = full["lev"].cast(pl.Float32).to_numpy()
    feats["code_affix"] = full["affix"].cast(pl.Float32).to_numpy()
    feats["code_subst1"] = full["subst1"].cast(pl.Float32).to_numpy()
    return pl.concat([d.select(ID_COLS), pl.DataFrame(feats)], how="horizontal")


def enrich(work: Path, split: str) -> None:
    q, s1 = _records(work, split)
    vocab = (
        s1.select(pl.col("nm_tok").list.unique().alias("t")).explode("t").drop_nulls()
        .group_by("t").agg(pl.len().alias("df"))
    )
    out_dir = work / split / "extra"
    out_dir.mkdir(exist_ok=True)
    for p in sorted((work / split / "parts").glob("*.parquet")):
        pairs = pl.read_parquet(p, columns=ID_COLS)
        pair_diff_features(pairs, q, s1, vocab).write_parquet(out_dir / p.name)
        _log(f"enriched {split}/{p.name}: {len(pairs):,} pairs")

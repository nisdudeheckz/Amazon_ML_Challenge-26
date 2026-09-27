"""Pairwise features for (S2/S3 query record, S1 candidate) pairs.

All features are country-agnostic (the country label itself is never a feature, so
unseen countries such as France are handled by the same model).
"""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from .vectors import NAMESPACES, NsCSR, pair_cosines

LEGAL = ["pvt", "ltd", "llc", "llp", "inc", "corp", "co", "pc", "lp", "plc", "pllc", "public",
         "sarl", "sas", "sa", "eurl", "sci", "snc"]


def _cp(a, b, scorer) -> np.ndarray:
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _legal_mask(nm: pl.Series) -> np.ndarray:
    """Bitmask of legal-form tokens present in a cleaned full name."""
    toks = nm.str.split(" ")
    m = np.zeros(len(nm), np.int64)
    for i, lg in enumerate(LEGAL):
        m |= toks.list.contains(lg).to_numpy().astype(np.int64) << i
    return m


def record_meta(df: pl.DataFrame) -> pl.DataFrame:
    """Per-record columns reused by the pair features."""
    return df.select(
        pl.col("nm_core").fill_null(""),
        pl.col("nm_concat").fill_null(""),
        pl.col("nm").fill_null(""),
        pl.col("ad").fill_null(""),
        pl.col("ad_first_num").fill_null(""),
        pl.col("nm_parts"),
        pl.col("ad_comp").list.len().cast(pl.Int16).alias("n_comp"),
        pl.col("nm_tok").list.len().cast(pl.Int16).alias("n_nmtok"),
        pl.col("nm_native"),
    )


def house_number_features(qf: pl.Series, sf: pl.Series) -> dict[str, np.ndarray]:
    """Compare first (house) numbers. Distractors are often near-duplicates of a real entity
    on the same street with a nearby but different number; true matches carry typo-like
    changes (dropped / swapped digit, zero padding)."""
    both = ((qf != "") & (sf != "")).to_numpy()
    lev = _cp(qf.to_list(), sf.to_list(), Levenshtein.distance)
    qn = qf.str.slice(0, 9).cast(pl.Int64, strict=False).to_numpy().astype(np.float64)
    sn = sf.str.slice(0, 9).cast(pl.Int64, strict=False).to_numpy().astype(np.float64)
    affix = (
        pl.DataFrame({"a": qf, "b": sf}).select(
            pl.col("a").str.starts_with(pl.col("b")) | pl.col("b").str.starts_with(pl.col("a"))
            | pl.col("a").str.ends_with(pl.col("b")) | pl.col("b").str.ends_with(pl.col("a"))
        ).to_series().to_numpy()
    )
    lendiff = (qf.str.len_chars().cast(pl.Int32) - sf.str.len_chars().cast(pl.Int32)).to_numpy()
    nan = np.float32("nan")
    return {
        "hn_lev": np.where(both, lev, nan).astype(np.float32),
        "hn_logdiff": np.where(both, np.log1p(np.abs(qn - sn)), nan).astype(np.float32),
        "hn_affix": np.where(both, affix, nan).astype(np.float32),
        "hn_lendiff": np.where(both, lendiff, nan).astype(np.float32),
    }


def pair_features(qi: np.ndarray, si: np.ndarray, q: pl.DataFrame, s1: pl.DataFrame,
                  qcsr: NsCSR, scsr: NsCSR, q_name_freq: np.ndarray, s1_name_freq: np.ndarray,
                  q_unseen: dict[str, np.ndarray] | None = None) -> pl.DataFrame:
    """Features for aligned pairs (q[qi], s1[si]).

    q / s1 are `record_meta` frames; *_name_freq give, per record, how many S1 records
    of the same country share its exact core name; q_unseen gives, per namespace, the
    fraction of a query's features that no S1 record has.
    """
    f: dict[str, np.ndarray] = {}
    # --- sparse namespace cosines / overlaps / jaccards
    cos = pair_cosines(qcsr, scsr, qi, si)
    f.update(cos)
    for ns in NAMESPACES:
        na = qcsr.nnz(ns)[qi].astype(np.float32)
        nb = scsr.nnz(ns)[si].astype(np.float32)
        ov = cos[f"ovl_{ns}"].astype(np.float32)
        f[f"jac_{ns}"] = np.where(na + nb - ov > 0, ov / np.maximum(na + nb - ov, 1), np.nan).astype(np.float32)
        f[f"q_nnz_{ns}"] = na
        f[f"s_nnz_{ns}"] = nb
    # --- strings
    qn = q["nm_core"].gather(qi).to_list()
    sn = s1["nm_core"].gather(si).to_list()
    f["nm_ratio"] = _cp(qn, sn, fuzz.ratio)
    f["nm_tsort"] = _cp(qn, sn, fuzz.token_sort_ratio)
    f["nm_tset"] = _cp(qn, sn, fuzz.token_set_ratio)
    f["nm_partial"] = _cp(qn, sn, fuzz.partial_ratio)
    qc = q["nm_concat"].gather(qi).to_list()
    sc = s1["nm_concat"].gather(si).to_list()
    f["nm_jw_concat"] = _cp(qc, sc, JaroWinkler.normalized_similarity)
    f["nm_concat_eq"] = (np.array(qc, dtype=object) == np.array(sc, dtype=object)).astype(np.int8)
    f["nm_full_ratio"] = _cp(q["nm"].gather(qi).to_list(), s1["nm"].gather(si).to_list(), fuzz.ratio)
    # best alias part of the query name vs S1 core name
    parts = pl.DataFrame({"pid": np.arange(len(qi), dtype=np.uint32), "p": q["nm_parts"].gather(qi), "s": sn})
    parts = parts.with_columns(pl.col("p").list.len().alias("np")).explode("p", empty_as_null=True).with_columns(pl.col("p").fill_null(""))
    pr = _cp(parts["p"].to_list(), parts["s"].to_list(), fuzz.ratio)
    ps = _cp(parts["p"].to_list(), parts["s"].to_list(), fuzz.token_set_ratio)
    pid = parts["pid"].to_numpy()
    best_r = np.zeros(len(qi), np.float32)
    best_s = np.zeros(len(qi), np.float32)
    np.maximum.at(best_r, pid, pr)
    np.maximum.at(best_s, pid, ps)
    f["nm_part_ratio"] = best_r
    f["nm_part_tset"] = best_s
    f["q_nparts"] = q["nm_parts"].list.len().gather(qi).to_numpy().astype(np.int8)
    qa = q["ad"].gather(qi).to_list()
    sa = s1["ad"].gather(si).to_list()
    f["ad_ratio"] = _cp(qa, sa, fuzz.ratio)
    f["ad_tsort"] = _cp(qa, sa, fuzz.token_sort_ratio)
    f["ad_tset"] = _cp(qa, sa, fuzz.token_set_ratio)
    f["ad_partial"] = _cp(qa, sa, fuzz.partial_ratio)
    # --- house / first numbers
    qf = q["ad_first_num"].gather(qi)
    sf = s1["ad_first_num"].gather(si)
    both = (qf != "") & (sf != "")
    f["fnum_eq"] = np.where(both.to_numpy(), (qf == sf).to_numpy().astype(np.float32), np.nan).astype(np.float32)
    f["fnum_jw"] = np.where(both.to_numpy(), _cp(qf.to_list(), sf.to_list(), JaroWinkler.normalized_similarity), np.nan).astype(np.float32)
    f.update(house_number_features(qf, sf))
    for k, v in (q_unseen or {}).items():
        f[f"q_{k}"] = v[qi]
    # --- legal forms
    ql = _legal_mask(q["nm"])[qi]
    sl = _legal_mask(s1["nm"])[si]
    f["legal_both"] = ((ql != 0) & (sl != 0)).astype(np.int8)
    f["legal_same"] = np.where((ql != 0) & (sl != 0), ((ql & sl) != 0).astype(np.float32), np.nan).astype(np.float32)
    # --- record meta
    f["q_nmlen"] = q["nm_core"].str.len_chars().gather(qi).to_numpy().astype(np.int16)
    f["s_nmlen"] = s1["nm_core"].str.len_chars().gather(si).to_numpy().astype(np.int16)
    f["q_adlen"] = q["ad"].str.len_chars().gather(qi).to_numpy().astype(np.int16)
    f["s_adlen"] = s1["ad"].str.len_chars().gather(si).to_numpy().astype(np.int16)
    f["q_ncomp"] = q["n_comp"].gather(qi).to_numpy()
    f["s_ncomp"] = s1["n_comp"].gather(si).to_numpy()
    f["q_nmtok"] = q["n_nmtok"].gather(qi).to_numpy()
    f["s_nmtok"] = s1["n_nmtok"].gather(si).to_numpy()
    f["q_native"] = q["nm_native"].gather(qi).to_numpy().astype(np.int8)
    f["q_name_freq"] = q_name_freq[qi].astype(np.float32)
    f["s_name_freq"] = s1_name_freq[si].astype(np.float32)
    return pl.DataFrame(f)

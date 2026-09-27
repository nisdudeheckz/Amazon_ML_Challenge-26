"""Hashed sparse TF-IDF representations of records, shared by blocking and features.

Namespaces (each gets its own idf-weighted, L2-normalised sub-vector):
    nw  name word tokens (all alias parts)
    ng  character 3-grams of the space-less core name
    aw  address alpha tokens
    an  address numeric tokens
    ac  address unit / plot codes ("203q", "b58", numbers)
    ap  (house number, street word) pairs
"""
from __future__ import annotations

import numba as nb
import numpy as np
import polars as pl

NAMESPACES = ("nw", "ng", "aw", "an", "ac", "ap")


def _ngrams(expr: pl.Expr, n: int = 3, max_len: int = 40) -> pl.Expr:
    grams = [expr.str.slice(k, n) for k in range(max_len - n + 1)]
    return pl.concat_list(grams).list.eval(pl.element().filter(pl.element().str.len_chars() == n)).list.unique()


def feature_table(df: pl.DataFrame) -> pl.DataFrame:
    """Long table (row u32, ns cat, h u64) of hashed features of a normalised frame."""
    e = pl.element()
    d = df.select(
        pl.int_range(pl.len(), dtype=pl.UInt32).alias("row"),
        pl.col("nm_parts").list.join(" ").str.split(" ").list.eval(e.filter(e.str.len_chars() > 0)).list.unique().alias("nw"),
        _ngrams(pl.col("nm_concat")).alias("ng"),
        pl.col("ad_tok").alias("aw"),
        pl.col("ad_num").alias("an"),
        pl.col("ad_code").alias("ac"),
        pl.when(pl.col("ad_first_num").is_not_null())
        .then(pl.col("ad_tok").list.slice(0, 2).list.eval(pl.lit("|") + e))
        .otherwise(pl.lit([], dtype=pl.List(pl.Utf8)))
        .alias("_st"),
        pl.col("ad_first_num").fill_null(""),
    )
    parts = []
    for ns in ("nw", "ng", "aw", "an", "ac"):
        parts.append(
            d.select("row", pl.col(ns).alias("f")).explode("f", empty_as_null=True).drop_nulls("f")
            .filter(pl.col("f") != "")
            .select("row", pl.lit(ns).alias("ns"), (pl.lit(ns + ":") + pl.col("f")).hash(7).alias("h"))
        )
    parts.append(
        d.select("row", "ad_first_num", pl.col("_st").alias("f")).explode("f", empty_as_null=True).drop_nulls("f")
        .select("row", pl.lit("ap").alias("ns"), (pl.lit("ap:") + pl.col("ad_first_num") + pl.col("f")).hash(7).alias("h"))
    )
    out = pl.concat(parts).unique(["row", "h"])
    return out.with_columns(pl.col("ns").cast(pl.Enum(list(NAMESPACES))))


def idf_table(ft: pl.DataFrame, n_docs: int) -> pl.DataFrame:
    df = ft.group_by("h").agg(pl.len().cast(pl.UInt32).alias("df"))
    return df.with_columns((np.log((n_docs + 1) / (pl.col("df") + 1)) + 1.0).cast(pl.Float32).alias("idf"))


def weighted(ft: pl.DataFrame, idf: pl.DataFrame, n_docs: int) -> pl.DataFrame:
    """Attach idf weights, L2-normalised per (row, namespace). Unseen features get max idf."""
    max_idf = float(np.log(n_docs + 1) + 1.0)
    w = ft.join(idf.select("h", "idf", "df"), on="h", how="left").with_columns(
        pl.col("idf").fill_null(max_idf), pl.col("df").fill_null(0)
    )
    return w.with_columns(
        (pl.col("idf") / (pl.col("idf") ** 2).sum().over(["row", "ns"]).sqrt()).alias("w")
    )


class NsCSR:
    """Per-namespace CSR (row -> sorted hashed feature ids, weights) for exact pair cosines."""

    def __init__(self, w: pl.DataFrame, n_rows: int):
        self.n_rows = n_rows
        self.mats = {}
        w = w.select("row", "ns", "h", "w").sort(["ns", "row", "h"])
        for ns in NAMESPACES:
            sub = w.filter(pl.col("ns") == ns)
            rows = sub["row"].to_numpy()
            counts = np.bincount(rows, minlength=n_rows) if len(rows) else np.zeros(n_rows, np.int64)
            indptr = np.zeros(n_rows + 1, np.int64)
            np.cumsum(counts, out=indptr[1:])
            self.mats[ns] = (indptr, sub["h"].to_numpy(), sub["w"].to_numpy().astype(np.float32))

    def nnz(self, ns: str) -> np.ndarray:
        return np.diff(self.mats[ns][0])


@nb.njit(parallel=True, cache=True)
def _pair_dots(ap, ai, ad, bp, bi, bd, qa, qb):
    n = qa.shape[0]
    out = np.zeros(n, np.float32)
    for t in nb.prange(n):
        i, j = qa[t], qb[t]
        x, xe = ap[i], ap[i + 1]
        y, ye = bp[j], bp[j + 1]
        s = 0.0
        while x < xe and y < ye:
            if ai[x] == bi[y]:
                s += ad[x] * bd[y]
                x += 1
                y += 1
            elif ai[x] < bi[y]:
                x += 1
            else:
                y += 1
        out[t] = s
    return out


@nb.njit(parallel=True, cache=True)
def _pair_overlap(ap, ai, bp, bi, qa, qb):
    """Number of shared features per pair."""
    n = qa.shape[0]
    out = np.zeros(n, np.int16)
    for t in nb.prange(n):
        i, j = qa[t], qb[t]
        x, xe = ap[i], ap[i + 1]
        y, ye = bp[j], bp[j + 1]
        c = 0
        while x < xe and y < ye:
            if ai[x] == bi[y]:
                c += 1
                x += 1
                y += 1
            elif ai[x] < bi[y]:
                x += 1
            else:
                y += 1
        out[t] = c
    return out


def pair_cosines(a: NsCSR, b: NsCSR, qa: np.ndarray, qb: np.ndarray) -> dict[str, np.ndarray]:
    """Cosine per namespace (+ shared-feature counts) for aligned row pairs (a[qa], b[qb])."""
    out = {}
    qa = qa.astype(np.int64)
    qb = qb.astype(np.int64)
    for ns in NAMESPACES:
        ap, ai, ad = a.mats[ns]
        bp, bi, bd = b.mats[ns]
        out[f"cos_{ns}"] = _pair_dots(ap, ai, ad, bp, bi, bd, qa, qb)
        out[f"ovl_{ns}"] = _pair_overlap(ap, ai, bp, bi, qa, qb)
    return out

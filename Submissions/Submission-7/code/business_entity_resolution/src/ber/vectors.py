"""Hashed sparse TF-IDF representations of records, shared by blocking and features.

Namespaces (each gets its own idf-weighted, L2-normalised sub-vector):
    nw  name word tokens (all alias parts)
    ng  character 3-grams of the space-less core name
    aw  address alpha tokens
    an  address numeric tokens
    ac  address unit / plot codes ("203q", "b58", numbers)
    ap  (house number, street word) pairs
  compound namespaces (combinations of common tokens are rare, so they survive the
  document-frequency pruning of the retrieval product):
    np  unordered pairs of core name tokens ("power|prime"), robust to word shuffles
    xn  core name token x address token ("apex|brooklyn"), a name + city key
    ab  adjacent address-token pairs ("grove|lake")
"""
from __future__ import annotations

import numba as nb
import numpy as np
import polars as pl

NAMESPACES = ("nw", "ng", "aw", "an", "ac", "ap", "np", "xn", "ab")


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
    nt = _long(df, "nm_tok", 6)
    at = _long(df, "ad_tok", 8)
    pairs = nt.join(nt, on="row", suffix="_b").filter(pl.col("pos") < pl.col("pos_b"))
    parts.append(pairs.select(
        "row", pl.lit("np").alias("ns"),
        (pl.lit("np:") + pl.min_horizontal("t", "t_b") + "|" + pl.max_horizontal("t", "t_b")).hash(7).alias("h")))
    cross = nt.filter(pl.col("pos") < 4).join(at.filter(pl.col("pos") < 6), on="row", suffix="_b")
    parts.append(cross.select("row", pl.lit("xn").alias("ns"), (pl.lit("xn:") + pl.col("t") + "|" + pl.col("t_b")).hash(7).alias("h")))
    adj = at.join(at.with_columns(pl.col("pos") - 1), on=["row", "pos"], suffix="_b")
    parts.append(adj.select("row", pl.lit("ab").alias("ns"), (pl.lit("ab:") + pl.col("t") + "|" + pl.col("t_b")).hash(7).alias("h")))
    out = pl.concat(parts).unique(["row", "h"])
    return out.with_columns(pl.col("ns").cast(pl.Enum(list(NAMESPACES))))


def _long(df: pl.DataFrame, col: str, cap: int) -> pl.DataFrame:
    """(row, pos, t) for the first `cap` tokens of a list column."""
    return (
        df.select(pl.int_range(pl.len(), dtype=pl.UInt32).alias("row"), pl.col(col).list.head(cap).alias("t"))
        .explode("t", empty_as_null=True).drop_nulls("t").filter(pl.col("t") != "")
        .with_columns(pl.int_range(pl.len(), dtype=pl.Int32).over("row").alias("pos"))
    )


NNS = len(NAMESPACES)


class Feats:
    """NumPy view of a feature table: row (u32), ns (namespace code, i8), h (u64)."""

    def __init__(self, ft: pl.DataFrame, n_rows: int):
        self.n_rows = n_rows
        self.row = ft["row"].to_numpy()
        self.ns = ft["ns"].to_physical().to_numpy().astype(np.int8)
        self.h = ft["h"].to_numpy()


class Stats:
    """Statistics of the features present in S1, as arrays sorted by hash.

    df  = S1 document frequency (drives pruning of the retrieval product);
    idf = over the whole unlabelled corpus of the country (S1+S2+S3), so tokens that are
          common in some source but rare in S1 (French departments vs. regions, US state
          names vs. codes) are not over-weighted. Features absent from S1 can never be
          matched by a candidate and are excluded from all vectors.
    """

    def __init__(self, s1: Feats):
        self.h, df = np.unique(s1.h, return_counts=True)
        self.df = df.astype(np.int64)
        self.cdf = self.df.copy()
        self.n_corpus = s1.n_rows

    def lookup(self, h: np.ndarray) -> np.ndarray:
        """Index of each hash in the sorted table, -1 if absent."""
        pos = np.searchsorted(self.h, h).astype(np.int32)
        pos[pos == len(self.h)] = 0
        return np.where(self.h[pos] == h, pos, np.int32(-1))

    def add_corpus(self, f: Feats) -> None:
        idx = self.lookup(f.h)
        self.cdf += np.bincount(idx[idx >= 0], minlength=len(self.h))
        self.n_corpus += f.n_rows

    def finalize(self, max_df: int) -> None:
        self.idf = (np.log((self.n_corpus + 1) / (self.cdf + 1)) + 1.0).astype(np.float32)
        keep = self.df <= max_df
        self.col = np.where(keep, np.cumsum(keep) - 1, -1).astype(np.int32)
        self.n_cols = int(keep.sum())


def weigh(f: Feats, stats: Stats) -> tuple[np.ndarray, ...]:
    """(row, ns, h, idx, w) of the S1-known features, idf-weighted and L2-normalised per
    (row, namespace)."""
    idx = stats.lookup(f.h)
    m = idx >= 0
    row, ns, h, idx = f.row[m], f.ns[m], f.h[m], idx[m]
    w = stats.idf[idx]
    g = row.astype(np.int64) * NNS + ns.astype(np.int64)
    norm = np.sqrt(np.bincount(g, weights=w.astype(np.float64) ** 2, minlength=f.n_rows * NNS))
    return row, ns, h, idx, (w / norm[g]).astype(np.float32)


def unseen_fraction(f: Feats, stats: Stats) -> dict[str, np.ndarray]:
    """Per namespace: fraction of a record's features that no S1 record has (NaN if none)."""
    g = f.row.astype(np.int64) * NNS + f.ns.astype(np.int64)
    total = np.bincount(g, minlength=f.n_rows * NNS)
    seen = np.bincount(g, weights=(stats.lookup(f.h) >= 0).astype(np.float64), minlength=f.n_rows * NNS)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(total > 0, 1.0 - seen / total, np.nan).astype(np.float32).reshape(f.n_rows, NNS)
    return {f"unseen_{ns}": frac[:, k].copy() for k, ns in enumerate(NAMESPACES)}


class NsCSR:
    """Per-namespace CSR (row -> sorted hashed feature ids, weights) for exact pair cosines."""

    def __init__(self, row: np.ndarray, ns: np.ndarray, h: np.ndarray, w: np.ndarray, n_rows: int):
        self.n_rows = n_rows
        self.mats = {}
        order = np.lexsort((h, row, ns))
        row, ns, h, w = row[order], ns[order], h[order], w[order]
        bounds = np.searchsorted(ns, np.arange(NNS + 1))
        for k, name in enumerate(NAMESPACES):
            a, b = bounds[k], bounds[k + 1]
            indptr = np.zeros(n_rows + 1, np.int64)
            np.cumsum(np.bincount(row[a:b], minlength=n_rows), out=indptr[1:])
            self.mats[name] = (indptr, h[a:b], w[a:b])

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

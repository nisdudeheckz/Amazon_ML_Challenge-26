"""Candidate generation: TF-IDF retrieval of Source-1 entities for every S2/S3 record.

Each S2/S3 record matches at most one S1 entity (verified on train: 7.64M matched ids,
all distinct), so blocking runs from the S2/S3 side: for every S2/S3 record we retrieve
the most similar S1 records carrying the same country label, scoring

    score_W = sum_ns  W[ns] * cos_ns(query, s1)

over the namespaces of `vectors.py`. The S1 matrix stores per-namespace L2-normalised
idf weights; the namespace weights W are applied on the query side, so one index serves
several weightings. Three retrievals are unioned:

    combined (name + address)  top-20   -> also defines the context features
    address-only               top-5     -> rescues junk / trade-name / transliterated names
    name-only                  top-5     -> rescues truncated / re-numbered addresses

Features whose S1 document frequency exceeds `max_df` are dropped from the sparse
product (they still count in the vector norms), so scores are lower bounds of the full
weighted cosines and the product stays cheap. The multiplication + top-K selection is
done by `sparse_dot_topn` (multi-threaded C++).
"""
from __future__ import annotations

import numpy as np
import polars as pl
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

from .vectors import NsCSR, feature_table, idf_table, weighted

# Weights tuned for recall@K on train.
W_COMBINED = {"nw": 0.25, "ng": 0.20, "aw": 0.30, "an": 0.0, "ac": 0.10, "ap": 0.15}
W_ADDRESS = {"aw": 0.50, "ac": 0.20, "ap": 0.30}
W_NAME = {"nw": 0.55, "ng": 0.45}


class S1Index:
    """Index of the Source-1 records of one country."""

    def __init__(self, s1: pl.DataFrame, max_df: int = 3000):
        self.n = len(s1)
        ft = feature_table(s1)
        self.idf = idf_table(ft, self.n)
        w = weighted(ft, self.idf, self.n)
        del ft
        self.csr = NsCSR(w, self.n)           # exact per-namespace vectors (features)
        self.vocab = (
            self.idf.filter(pl.col("df") <= max_df).select("h").with_row_index("col")
        )
        w = w.join(self.vocab, on="h", how="inner")
        self.mat_t = sp.csr_matrix(
            (w["w"].to_numpy(), (w["col"].to_numpy(), w["row"].to_numpy())),
            shape=(len(self.vocab), self.n), dtype=np.float32,
        )

    def encode(self, q: pl.DataFrame) -> tuple[pl.DataFrame, NsCSR]:
        """Weighted feature table (restricted to the product vocabulary) + per-namespace
        CSR (full vocabulary) of query records."""
        w = weighted(feature_table(q), self.idf, self.n)
        csr = NsCSR(w, len(q))
        return w.join(self.vocab, on="h", how="inner").select("row", "ns", "col", "w"), csr

    def topk(self, qw: pl.DataFrame, n_q: int, k: int, weights: dict[str, float],
             n_threads: int = 8) -> pl.DataFrame:
        """(q_row, s1_row, bscore, brank) for the top-k S1 rows of each query row."""
        scale = pl.col("ns").cast(pl.Utf8).replace_strict(weights, default=0.0, return_dtype=pl.Float32)
        w = qw.with_columns((pl.col("w") * scale).alias("w")).filter(pl.col("w") > 0)
        qm = sp.csr_matrix(
            (w["w"].to_numpy(), (w["row"].to_numpy(), w["col"].to_numpy())),
            shape=(n_q, self.mat_t.shape[0]), dtype=np.float32,
        )
        res = sp_matmul_topn(qm, self.mat_t, top_n=k, sort=True, n_threads=n_threads).tocsr()
        cnt = np.diff(res.indptr)
        return pl.DataFrame({
            "q_row": np.repeat(np.arange(n_q, dtype=np.uint32), cnt),
            "s1_row": res.indices.astype(np.uint32),
            "bscore": res.data.astype(np.float32),
            "brank": (np.arange(res.nnz) - np.repeat(res.indptr[:-1], cnt)).astype(np.uint8),
        })

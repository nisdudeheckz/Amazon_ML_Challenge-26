"""Candidate generation: TF-IDF retrieval of Source-1 entities for every S2/S3 record.

Each S2/S3 record matches at most one S1 entity (verified on train: 7.64M matched ids,
all distinct), so blocking runs from the S2/S3 side: for every S2/S3 record we retrieve
the most similar S1 records carrying the same country label, scoring

    score_W = sum_ns  W[ns] * cos_ns(query, s1)

over the namespaces of `vectors.py`. The S1 matrix stores per-namespace L2-normalised
idf weights; the namespace weights W are applied on the query side. (Unioning extra
address-only / name-only retrievals was tried: +0.1-0.4 pt recall for 2-3x candidates,
so a single combined retrieval is used.) The top-20 are retrieved (context features)
and pruned to rank < 10 and score >= 0.6 x the record's best score.

Features whose S1 document frequency exceeds `max_df` are dropped from the sparse
product (they still count in the vector norms), so scores are lower bounds of the full
weighted cosines and the product stays cheap. Pruning alone lost many true pairs whose
evidence is a *combination* of common tokens (city + generic name words); the compound
namespaces (name-token pairs, name x address tokens, address bigrams) are rare, survive
the pruning and recover almost all of that recall. The multiplication + top-K selection is
done by `sparse_dot_topn` (multi-threaded C++).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import polars as pl
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

from .vectors import NAMESPACES, Feats, NsCSR, Stats, feature_table, unseen_fraction, weigh

# Weights tuned for recall on 60k-record train samples (pruned-candidate recall with
# max_df=6000: US 98.8%, India 98.1%; without the compound namespaces and max_df=3000:
# US 96.4%, India 93.6%).
W_COMBINED = {"nw": 0.15, "ng": 0.15, "aw": 0.20, "an": 0.0, "ac": 0.10, "ap": 0.10,
              "np": 0.10, "xn": 0.15, "ab": 0.05}
MAX_DF = 6000


class S1Index:
    """Index of the Source-1 records of one country.

    `corpus` yields further (unlabelled) record frames of the same country — the S2/S3
    records — whose document frequencies enter the idf; without it idf comes from S1.
    """

    def __init__(self, s1: pl.DataFrame, corpus: Iterable[pl.DataFrame] = (), max_df: int = MAX_DF):
        self.n = len(s1)
        f = Feats(feature_table(s1), self.n)
        self.stats = Stats(f)
        for q in corpus:
            self.stats.add_corpus(Feats(feature_table(q), len(q)))
        self.stats.finalize(max_df)
        row, ns, h, idx, w = weigh(f, self.stats)
        del f
        self.csr = NsCSR(row, ns, h, w, self.n)          # exact per-namespace vectors (features)
        col = self.stats.col[idx]
        m = col >= 0
        self.mat_t = sp.csr_matrix((w[m], (col[m], row[m])), shape=(self.stats.n_cols, self.n), dtype=np.float32)

    def encode(self, q: pl.DataFrame) -> tuple[tuple[np.ndarray, ...], NsCSR, dict[str, np.ndarray]]:
        """Product-vocabulary entries (row, ns, col, w), per-namespace CSR (all S1-known
        features) and unseen-feature fractions of query records."""
        f = Feats(feature_table(q), len(q))
        unseen = unseen_fraction(f, self.stats)
        row, ns, h, idx, w = weigh(f, self.stats)
        del f
        csr = NsCSR(row, ns, h, w, len(q))
        col = self.stats.col[idx]
        m = col >= 0
        return (row[m], ns[m], col[m], w[m]), csr, unseen

    def topk(self, qv: tuple[np.ndarray, ...], n_q: int, k: int, weights: dict[str, float],
             n_threads: int = 8) -> pl.DataFrame:
        """(q_row, s1_row, bscore, brank) for the top-k S1 rows of each query row."""
        row, ns, col, w = qv
        wv = np.array([weights.get(n, 0.0) for n in NAMESPACES], np.float32)[ns] * w
        m = wv > 0
        qm = sp.csr_matrix((wv[m], (row[m], col[m])), shape=(n_q, self.stats.n_cols), dtype=np.float32)
        res = sp_matmul_topn(qm, self.mat_t, top_n=k, sort=True, n_threads=n_threads).tocsr()
        cnt = np.diff(res.indptr)
        return pl.DataFrame({
            "q_row": np.repeat(np.arange(n_q, dtype=np.uint32), cnt),
            "s1_row": res.indices.astype(np.uint32),
            "bscore": res.data.astype(np.float32),
            "brank": (np.arange(res.nnz) - np.repeat(res.indptr[:-1], cnt)).astype(np.uint8),
        })

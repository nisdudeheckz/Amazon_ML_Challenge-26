"""Macro F0.5 exactly as described in the challenge (singletons included)."""
from __future__ import annotations

import polars as pl


def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series, beta: float = 0.5) -> dict:
    """pred / truth: long (s1_id, rid) pairs. s1_ids: the S1 entities to average over.

    Per entity: F = (1+b^2) TP / ((1+b^2) TP + b^2 FN + FP); 1.0 if both sets are empty.
    """
    b2 = beta * beta
    base = pl.DataFrame({"s1_id": s1_ids})
    p = pred.select("s1_id", "rid").unique().filter(pl.col("s1_id").is_in(s1_ids.implode()))
    t = truth.select("s1_id", "rid").unique().filter(pl.col("s1_id").is_in(s1_ids.implode()))
    tp = p.join(t, on=["s1_id", "rid"]).group_by("s1_id").agg(pl.len().alias("tp"))
    np_ = p.group_by("s1_id").agg(pl.len().alias("np"))
    nt = t.group_by("s1_id").agg(pl.len().alias("nt"))
    d = (
        base.join(tp, on="s1_id", how="left").join(np_, on="s1_id", how="left").join(nt, on="s1_id", how="left")
        .fill_null(0)
        .with_columns(
            pl.when((pl.col("np") == 0) & (pl.col("nt") == 0)).then(1.0)
            .otherwise((1 + b2) * pl.col("tp") / ((1 + b2) * pl.col("tp") + b2 * (pl.col("nt") - pl.col("tp")) + (pl.col("np") - pl.col("tp"))))
            .alias("f")
        )
    )
    single = d.filter(pl.col("nt") == 0)
    multi = d.filter(pl.col("nt") > 0)
    tot_tp, tot_p, tot_t = d["tp"].sum(), d["np"].sum(), d["nt"].sum()
    return {
        "f05": float(d["f"].mean()),
        "f05_singletons": float(single["f"].mean()) if len(single) else float("nan"),
        "f05_nonsingletons": float(multi["f"].mean()) if len(multi) else float("nan"),
        "pair_precision": tot_tp / tot_p if tot_p else 0.0,
        "pair_recall": tot_tp / tot_t if tot_t else 0.0,
        "n": len(d),
    }

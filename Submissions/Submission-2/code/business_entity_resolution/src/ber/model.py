"""Pair classifier (LightGBM), decision rule and train / predict steps.

Validation protocol: 5% of the training S1 entities are held out. Every query that has
*any* held-out S1 among its candidates is excluded from model fitting, so the predicted
match sets of held-out S1 entities are complete and fully out-of-sample, and macro F0.5
is computed on them exactly as on the leaderboard (singletons included).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from . import candidates
from .io import read_ground_truth, read_source, write_id_lists
from .metrics import macro_f05

ID_COLS = ["q_id", "s1_id"]
EVAL_FRAC = 50     # per mille of S1 entities held out
TRAIN_FRAC = 400   # per mille of remaining queries used for fitting
PARAMS = dict(
    objective="binary", learning_rate=0.08, num_leaves=255, min_data_in_leaf=200,
    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    max_bin=255, num_threads=16, verbose=-1, seed=7,
)
NUM_ROUNDS = 1500


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _eval_s1(expr: pl.Expr) -> pl.Expr:
    return (expr.hash(11) % 1000) < EVAL_FRAC


def _train_q(expr: pl.Expr) -> pl.Expr:
    return (expr.hash(13) % 1000) < TRAIN_FRAC


def context_path(work: Path, split: str) -> Path:
    return work / split / "s1ctx.parquet"


def ensure_context(work: Path, split: str) -> Path:
    p = context_path(work, split)
    if not p.exists():
        candidates.s1_context(work / split / "parts").write_parquet(p)
    return p


def scan_pairs(work: Path, split: str) -> pl.LazyFrame:
    parts = pl.scan_parquet(str(work / split / "parts" / "*.parquet"))
    ctx = pl.scan_parquet(ensure_context(work, split))
    return parts.join(ctx, on=ID_COLS, how="left")


def feature_names(lf: pl.LazyFrame) -> list[str]:
    return [c for c in lf.collect_schema().names() if c not in ID_COLS]


def _to_numpy(df: pl.DataFrame, feats: list[str]) -> np.ndarray:
    return df.select(pl.col(feats).cast(pl.Float32)).to_numpy()


# ----------------------------------------------------------------------------- decision

def assign(pairs: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """Each S2/S3 record goes to its single most probable S1 candidate if p >= threshold.

    pairs: (q_id, s1_id, p). Returns long (s1_id, rid).
    """
    best = pairs.sort("p", descending=True).group_by("q_id", maintain_order=True).first()
    return best.filter(pl.col("p") >= threshold).select("s1_id", pl.col("q_id").alias("rid"))


def assign_expected_f(pairs: pl.DataFrame, lam: float = 0.3, floor: float = 0.05) -> pl.DataFrame:
    """Per-S1 expected-F0.5 maximisation on top of the per-record argmax.

    For an S1 entity whose assigned records have probabilities p_1 >= p_2 >= ... the
    expected F0.5 of keeping the top m is approximated by
        1.25 * sum_{i<=m} p_i / (m + 0.25 * (sum_i p_i + lam))
    (lam ~ expected number of true matches the candidates missed), and keeping none
    scores P(no true match) = prod_i (1 - p_i) * exp(-lam). The best m is kept.
    """
    best = pairs.sort("p", descending=True).group_by("q_id", maintain_order=True).first()
    g = (
        best.filter(pl.col("p") >= floor)
        .sort(["s1_id", "p"], descending=[False, True])
        .with_columns(
            pl.col("p").cum_sum().over("s1_id").alias("cum_p"),
            pl.int_range(1, pl.len() + 1).over("s1_id").alias("m"),
            pl.col("p").sum().over("s1_id").alias("tot"),
            ((1 - pl.col("p")).clip(1e-6, 1).log().sum().over("s1_id") - lam).exp().alias("score0"),
        )
        .with_columns((1.25 * pl.col("cum_p") / (pl.col("m") + 0.25 * (pl.col("tot") + lam))).alias("score_m"))
    )
    g = g.with_columns(
        pl.col("score_m").max().over("s1_id").alias("best_score"),
        pl.col("m").filter(pl.col("score_m") == pl.col("score_m").max()).first().over("s1_id").alias("best_m"),
    )
    keep = g.filter((pl.col("best_score") > pl.col("score0")) & (pl.col("m") <= pl.col("best_m")))
    return keep.select("s1_id", pl.col("q_id").alias("rid"))


def tune_threshold(pairs: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series) -> tuple[float, dict, list]:
    grid = [round(x, 3) for x in np.arange(0.20, 0.96, 0.025)]
    rows = []
    for t in grid:
        m = macro_f05(assign(pairs, t), truth, s1_ids)
        rows.append((t, m))
    best_t, best_m = max(rows, key=lambda r: r[1]["f05"])
    return best_t, best_m, rows


def tune_expected_f(pairs: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series) -> tuple[tuple, dict, list]:
    """Grid over (lam, floor); the floor keeps lone weak candidates from being matched,
    which protects singleton S1 entities."""
    rows = []
    for floor in (0.05, 0.4, 0.5, 0.6, 0.65, 0.7):
        for lam in (0.0, 0.2, 0.5, 1.0):
            rows.append(((lam, floor), macro_f05(assign_expected_f(pairs, lam, floor), truth, s1_ids)))
    best_p, best_m = max(rows, key=lambda r: r[1]["f05"])
    return best_p, best_m, rows


def _fmt(m: dict) -> str:
    return (f"f05={m['f05']:.5f} single={m['f05_singletons']:.4f} multi={m['f05_nonsingletons']:.4f} "
            f"P={m['pair_precision']:.4f} R={m['pair_recall']:.4f}")


def choose_decision(ev: pl.DataFrame, truth: pl.DataFrame, s1_eval: pl.Series) -> dict:
    """Tune both decision rules on held-out S1 entities; keep the better one."""
    best_t, m_t, rows = tune_threshold(ev, truth, s1_eval)
    for t, m in rows:
        _log(f"  threshold={t:.3f} {_fmt(m)}")
    (best_l, best_f), m_l, rows = tune_expected_f(ev, truth, s1_eval)
    for (lam, floor), m in rows:
        _log(f"  expected-F lam={lam:.2f} floor={floor:.2f} {_fmt(m)}")
    _log(f"best threshold rule: t={best_t} {_fmt(m_t)}")
    _log(f"best expected-F rule: lam={best_l} floor={best_f} {_fmt(m_l)}")
    common = {"lam": best_l, "floor": best_f, "threshold": best_t,
              "metrics_threshold": m_t, "metrics_expected_f": m_l}
    if m_l["f05"] > m_t["f05"]:
        return {"rule": "expected_f", "metrics": m_l, **common}
    return {"rule": "threshold", "metrics": m_t, **common}


def decide(pairs: pl.DataFrame, report: dict) -> pl.DataFrame:
    if report["rule"] == "expected_f":
        return assign_expected_f(pairs, report["lam"], report.get("floor", 0.05))
    return assign(pairs, report["threshold"])


# ----------------------------------------------------------------------------- steps

def step_train(data: Path, work: Path) -> None:
    lf = scan_pairs(work, "train")
    feats = feature_names(lf)
    truth = read_ground_truth(data / "train" / "train_ground_truth.tsv").rename({"match_id": "rid"})
    labels = truth.select(pl.col("s1_id"), pl.col("rid").alias("q_id"), pl.lit(1, pl.Int8).alias("y"))

    eval_q = lf.filter(_eval_s1(pl.col("s1_id"))).select("q_id").unique().collect()
    _log(f"held-out queries: {len(eval_q):,}")
    tr = (
        lf.join(eval_q.lazy(), on="q_id", how="anti")
        .filter(_train_q(pl.col("q_id")))
        .join(labels.lazy(), on=ID_COLS, how="left")
        .with_columns(pl.col("y").fill_null(0))
        .collect()
    )
    _log(f"fit pairs: {len(tr):,} (positives {tr['y'].sum():,})")
    X, y = _to_numpy(tr, feats), tr["y"].to_numpy()
    del tr
    ev = lf.join(eval_q.lazy(), on="q_id", how="semi").join(labels.lazy(), on=ID_COLS, how="left") \
        .with_columns(pl.col("y").fill_null(0)).collect()
    Xe, ye = _to_numpy(ev, feats), ev["y"].to_numpy()
    _log(f"eval pairs: {len(ev):,} (positives {ye.sum():,})")

    dtr = lgb.Dataset(X, y, feature_name=feats, free_raw_data=True)
    dev = lgb.Dataset(Xe, ye, reference=dtr)
    booster = lgb.train(
        PARAMS, dtr, NUM_ROUNDS, valid_sets=[dev], valid_names=["eval"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    del X, dtr
    booster.save_model(str(work / "model_eval.txt"))
    ev = ev.select(ID_COLS).with_columns(pl.Series("p", booster.predict(Xe, num_threads=16)))
    ev.write_parquet(work / "eval_pred.parquet")

    s1 = read_source(data / "train" / "train_source1.tsv").select("entity_id")
    s1_eval = s1.filter(_eval_s1(pl.col("entity_id")))["entity_id"]
    decision = choose_decision(ev, truth, s1_eval)
    imp = sorted(zip(feats, booster.feature_importance("gain")), key=lambda r: -r[1])
    with open(work / "train_report.json", "w") as f:
        json.dump({**decision, "best_iteration": booster.best_iteration, "n_eval_s1": len(s1_eval),
                   "features": feats, "importance": [(a, float(b)) for a, b in imp]}, f, indent=1)

    # Refit on all training queries' sample (incl. held-out ones) with the tuned #rounds.
    full = (
        lf.filter(_train_q(pl.col("q_id")) | pl.col("q_id").is_in(eval_q["q_id"].implode()))
        .join(labels.lazy(), on=ID_COLS, how="left").with_columns(pl.col("y").fill_null(0)).collect()
    )
    _log(f"refit pairs: {len(full):,}")
    X, y = _to_numpy(full, feats), full["y"].to_numpy()
    del full
    final = lgb.train(PARAMS, lgb.Dataset(X, y, feature_name=feats), int(booster.best_iteration * 1.05))
    final.save_model(str(work / "model_final.txt"))
    _log("saved final model")


def step_tune(data: Path, work: Path) -> None:
    """Re-tune the decision rule from saved held-out predictions (no retraining)."""
    ev = pl.read_parquet(work / "eval_pred.parquet")
    truth = read_ground_truth(data / "train" / "train_ground_truth.tsv").rename({"match_id": "rid"})
    s1 = read_source(data / "train" / "train_source1.tsv").select("entity_id")
    s1_eval = s1.filter(_eval_s1(pl.col("entity_id")))["entity_id"]
    report = json.load(open(work / "train_report.json"))
    report.update(choose_decision(ev, truth, s1_eval))
    with open(work / "train_report.json", "w") as f:
        json.dump(report, f, indent=1)
    _log(f"decision rule: {report['rule']} -> f05={report['metrics']['f05']:.5f}")


def step_predict(data: Path, work: Path, out_dir: Path, model_name: str = "model_final.txt") -> None:
    report = json.load(open(work / "train_report.json"))
    booster = lgb.Booster(model_file=str(work / model_name))
    feats = booster.feature_name()
    ensure_context(work, "test")
    parts = sorted((work / "test" / "parts").glob("*.parquet"))
    ctx = pl.read_parquet(context_path(work, "test"))
    preds = []
    for p in parts:
        df = pl.read_parquet(p).join(ctx, on=ID_COLS, how="left")
        preds.append(df.select(ID_COLS).with_columns(pl.Series("p", booster.predict(_to_numpy(df, feats), num_threads=16))))
        _log(f"scored {p.name}: {len(df):,} pairs")
    pairs = pl.concat(preds)
    pairs.write_parquet(work / "test_pred.parquet")
    s1_ids = read_source(data / "test" / "test_source1.tsv")["entity_id"]
    matches = decide(pairs, report)
    write_id_lists(s1_ids, pairs.select("s1_id", pl.col("q_id").alias("rid")), "candidate_entity_ids",
                   out_dir / "candidate_pairs.tsv")
    write_id_lists(s1_ids, matches, "matched_entity_ids", out_dir / "matching_results.tsv")
    _log(f"wrote {out_dir}: {len(matches):,} matches for {matches['s1_id'].n_unique():,} of {len(s1_ids):,} S1 entities")

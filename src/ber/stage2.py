"""Stage 2: re-score pairs with features derived from stage-1 probabilities.

Stage 1 (the pair features of `candidates.py`) is fitted twice, on the two halves of the
fitting sample split by a hash of the S2/S3 id ("fold"). Every pair — train or test — is
scored by the model of the *other* fold, so stage-1 probabilities are out-of-fold on
train and identically distributed on test.

Stage-2 features = stage-1 features + p1 + competition features:
  record level: rank of p1 among the record's candidates, best / second-best p1,
                sum of p1, share and margin of this candidate
  S1 level:     how many other records claim the same S1 (all candidates, and only as
                their best candidate), sum / max of their p1, rank of this record

All data is processed part file by part file (`work/<split>/parts/*.parquet`); the
aggregates are written next to them in `work/<split>/s2feat/`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

import lightgbm as lgb
import numpy as np
import polars as pl

from . import config
from .enrich import enrich
from .io import read_ground_truth, read_source, write_id_lists
from .metrics import macro_f05
from .model import DROP_FEATURES, ID_COLS, PARAMS, _eval_s1, _train_q, choose_decision, decide, ensure_context

S1_ROUNDS = 1000
S2_ROUNDS = 2000


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _fold(expr: pl.Expr) -> pl.Expr:
    return (expr.hash(17) % 2).cast(pl.Int8)


def _labels(data: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    truth = read_ground_truth(data / "train" / "train_ground_truth.tsv").rename({"match_id": "rid"})
    labels = truth.select("s1_id", pl.col("rid").alias("q_id"), pl.lit(1, pl.Int8).alias("y"))
    return truth, labels


def _parts(work: Path, split: str) -> list[Path]:
    return sorted((work / split / "parts").glob("*.parquet"))


def _feats(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLS and c not in ("y", "_f") and c not in DROP_FEATURES]


def _X(df: pl.DataFrame, feats: list[str]) -> np.ndarray:
    return df.select(pl.col(feats).cast(pl.Float32)).to_numpy()


def _load_part(work: Path, split: str, p: Path, ctx: pl.DataFrame | None = None) -> pl.DataFrame:
    """A candidate part joined with the S1-side context and the `enrich` features."""
    df = pl.read_parquet(p)
    if ctx is not None:
        df = df.join(ctx, on=ID_COLS, how="left")
    extra = work / split / "extra" / p.name
    if extra.exists():
        df = df.join(pl.read_parquet(extra), on=ID_COLS, how="left")
    return df


def _eval_queries(work: Path) -> pl.DataFrame:
    """Records having any held-out S1 among their candidates (never used for fitting)."""
    return (
        pl.scan_parquet(work / "train" / "p1.parquet").filter(_eval_s1(pl.col("s1_id")))
        .select("q_id").unique().collect()
    )


# ----------------------------------------------------------------------------- stage 1

def fit_stage1_folds(data: Path, work: Path) -> None:
    _, labels = _labels(data)
    ctx = pl.read_parquet(ensure_context(work, "train"))
    # eval queries from the raw candidate lists (p1 does not exist yet)
    eval_q = (
        pl.scan_parquet(str(work / "train" / "parts" / "*.parquet")).select(ID_COLS)
        .filter(_eval_s1(pl.col("s1_id"))).select("q_id").unique().collect()
    )
    X: dict[int, list[np.ndarray]] = {0: [], 1: []}
    Y: dict[int, list[np.ndarray]] = {0: [], 1: []}
    feats: list[str] = []
    for p in _parts(work, "train"):
        df = (
            _load_part(work, "train", p, ctx).join(eval_q, on="q_id", how="anti").filter(_train_q(pl.col("q_id")))
            .join(labels, on=ID_COLS, how="left")
            .with_columns(pl.col("y").fill_null(0), _fold(pl.col("q_id")).alias("_f"))
        )
        feats = feats or _feats(df)
        for f in (0, 1):
            sub = df.filter(pl.col("_f") == f)
            X[f].append(_X(sub, feats))
            Y[f].append(sub["y"].to_numpy())
    del ctx
    for f in (0, 1):
        y = np.concatenate(Y[f])
        _log(f"stage-1 fold {f}: {len(y):,} pairs, {len(feats)} features")
        booster = lgb.train(PARAMS, lgb.Dataset(X[f], y, feature_name=feats, free_raw_data=True), S1_ROUNDS)
        booster.save_model(str(work / f"stage1_fold{f}.txt"))
        X[f] = []
        del booster


def predict_stage1(work: Path, split: str) -> None:
    """p1 for every pair of the split, each record scored by the other fold's model."""
    models = [lgb.Booster(model_file=str(work / f"stage1_fold{f}.txt")) for f in (0, 1)]
    feats = models[0].feature_name()
    ctx = pl.read_parquet(ensure_context(work, split))
    outs = []
    for p in _parts(work, split):
        df = _load_part(work, split, p, ctx).with_columns(_fold(pl.col("q_id")).alias("_f"))
        p1 = np.zeros(len(df), np.float32)
        for f in (0, 1):
            m = (df["_f"] == f).to_numpy()
            if m.any():
                p1[m] = models[1 - f].predict(_X(df.filter(pl.col("_f") == f), feats), num_threads=config.N_THREADS)
        outs.append(df.select(ID_COLS).with_columns(pl.Series("p1", p1), pl.lit(p.stem).alias("part")))
        _log(f"stage-1 scored {split}/{p.name}")
    pl.concat(outs).write_parquet(work / split / "p1.parquet")


# ----------------------------------------------------------------------------- stage-2 features

def build_stage2_features(work: Path, split: str) -> None:
    p1 = pl.read_parquet(work / split / "p1.parquet")
    agg = p1.with_columns(
        pl.col("p1").rank("ordinal", descending=True).over("q_id").cast(pl.Int16).alias("q_p1_rank"),
        pl.col("p1").max().over("q_id").alias("q_p1_max"),
        pl.col("p1").sum().over("q_id").alias("q_p1_sum"),
        pl.col("p1").sort(descending=True).get(1, null_on_oob=True).fill_null(0).over("q_id").alias("q_p1_2nd"),
        pl.col("p1").sum().over("s1_id").alias("s_p1_sum"),
        pl.col("p1").max().over("s1_id").alias("s_p1_max"),
        (pl.col("p1") > 0.5).sum().over("s1_id").cast(pl.Int16).alias("s_p1_n50"),
        pl.col("p1").rank("ordinal", descending=True).over("s1_id").cast(pl.Int16).alias("s_p1_rank"),
    ).with_columns(
        pl.when(pl.col("q_p1_rank") == 1).then(pl.col("q_p1_2nd")).otherwise(pl.col("q_p1_max")).alias("q_p1_other"),
        (pl.col("p1") / pl.col("q_p1_sum").clip(1e-6)).alias("q_p1_share"),
        (pl.col("s_p1_sum") - pl.col("p1")).alias("s_p1_others"),
        (pl.col("s_p1_n50") - (pl.col("p1") > 0.5).cast(pl.Int16)).alias("s_p1_n50_others"),
    ).with_columns((pl.col("p1") - pl.col("q_p1_other")).alias("q_p1_margin"))
    # claims counted only over each record's best candidate (what the assignment sees)
    best = agg.filter(pl.col("q_p1_rank") == 1).group_by("s1_id").agg(
        pl.col("p1").sum().alias("s_best_sum"), (pl.col("p1") > 0.5).sum().cast(pl.Int16).alias("s_best_n50"))
    agg = agg.join(best, on="s1_id", how="left").with_columns(pl.col("s_best_sum", "s_best_n50").fill_null(0))
    agg = agg.join(pl.read_parquet(ensure_context(work, split)), on=ID_COLS, how="left")
    out_dir = work / split / "s2feat"
    out_dir.mkdir(exist_ok=True)
    for (part,), sub in agg.partition_by("part", as_dict=True).items():
        sub.drop("part").write_parquet(out_dir / f"{part}.parquet")
    _log(f"stage-2 features for {split}: {agg.shape}")


def iter_stage2(work: Path, split: str) -> Iterator[pl.DataFrame]:
    for p in _parts(work, split):
        yield _load_part(work, split, p).join(pl.read_parquet(work / split / "s2feat" / p.name), on=ID_COLS, how="left")


# ----------------------------------------------------------------------------- stage-2 model

def _report_stage1(work: Path, truth: pl.DataFrame, s1_eval: pl.Series, eval_q: pl.DataFrame) -> dict:
    """Held-out metric of the out-of-fold stage-1 probabilities (for comparison)."""
    ev = pl.read_parquet(work / "train" / "p1.parquet").join(eval_q, on="q_id", how="semi") \
        .select(*ID_COLS, pl.col("p1").alias("p"))
    return choose_decision(ev, truth, s1_eval)


PSEUDO_POS = 0.97       # test record's best candidate at/above this -> pseudo-match
PSEUDO_NEG = 0.03       # test candidates at/below this -> pseudo-non-match ...
PSEUDO_NEG_FRAC = 300   # ... of which this per mille is kept


def pseudo_rows(work: Path, feats: list[str]) -> tuple[list[np.ndarray], np.ndarray]:
    """Self-training rows from the (unlabelled) test split: confident stage-2 predictions
    become labels. Same rule for every country; no test labels exist or are used."""
    p = pl.read_parquet(work / "test_pred_s2.parquet")
    best = pl.col("p") == pl.col("p").max().over("q_id")
    lab = pl.concat([
        p.filter(best & (pl.col("p") >= PSEUDO_POS)).select(*ID_COLS, pl.lit(1, pl.Int8).alias("y")),
        p.filter((pl.col("p") <= PSEUDO_NEG) & ((pl.col("q_id").hash(31) % 1000) < PSEUDO_NEG_FRAC))
        .select(*ID_COLS, pl.lit(0, pl.Int8).alias("y")),
    ])
    X, y = [], []
    for df in iter_stage2(work, "test"):
        d = df.join(lab, on=ID_COLS, how="inner")
        X.append(_X(d, feats))
        y.append(d["y"].to_numpy())
    y = np.concatenate(y)
    _log(f"self-training rows from test: {int(y.sum()):,} pseudo-matches, {int((y == 0).sum()):,} pseudo-non-matches")
    return X, y


def fit_stage2(data: Path, work: Path, extra: tuple[list[np.ndarray], np.ndarray] | None = None,
               tag: str = "") -> None:
    """Fit the stage-2 model (optionally with extra rows, e.g. test pseudo-labels), tune the
    decision rule on held-out training entities, save model + report with suffix `tag`."""
    truth, labels = _labels(data)
    eval_q = _eval_queries(work)
    Xtr, ytr, Xes, yes = [], [], [], []
    feats: list[str] = []
    for df in iter_stage2(work, "train"):
        feats = feats or _feats(df)
        tr = df.join(eval_q, on="q_id", how="anti").filter(_train_q(pl.col("q_id")))             .join(labels, on=ID_COLS, how="left").with_columns(pl.col("y").fill_null(0))
        # early stopping on a 30% subsample of held-out records
        es = df.join(eval_q, on="q_id", how="semi").filter((pl.col("q_id").hash(23) % 10) < 3)             .join(labels, on=ID_COLS, how="left").with_columns(pl.col("y").fill_null(0))
        Xtr.append(_X(tr, feats)); ytr.append(tr["y"].to_numpy())
        Xes.append(_X(es, feats)); yes.append(es["y"].to_numpy())
    ytr, yes = np.concatenate(ytr), np.concatenate(yes)
    if extra is not None:
        Xtr.extend(extra[0])
        ytr = np.concatenate([ytr, extra[1]])
    _log(f"stage-2{tag} fit pairs: {len(ytr):,}, early-stopping pairs: {len(yes):,}, {len(feats)} features")
    dtr = lgb.Dataset(Xtr, ytr, feature_name=feats, free_raw_data=True)
    dev = lgb.Dataset(np.concatenate(Xes), yes, reference=dtr)
    del Xtr, Xes
    booster = lgb.train(PARAMS, dtr, S2_ROUNDS, valid_sets=[dev], valid_names=["eval"],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    booster.save_model(str(work / f"stage2{tag}.txt"))
    del dtr, dev
    preds = []
    for df in iter_stage2(work, "train"):
        df = df.join(eval_q, on="q_id", how="semi")
        preds.append(df.select(ID_COLS).with_columns(pl.Series("p", booster.predict(_X(df, feats), num_threads=config.N_THREADS))))
    ev = pl.concat(preds)
    ev.write_parquet(work / f"eval_pred_s2{tag}.parquet")
    s1_eval = read_source(data / "train" / "train_source1.tsv").select("entity_id") \
        .filter(_eval_s1(pl.col("entity_id")))["entity_id"]
    _log("stage-1 (out-of-fold) held-out decision tuning:")
    s1_decision = _report_stage1(work, truth, s1_eval, eval_q)
    _log("stage-2 held-out decision tuning:")
    decision = choose_decision(ev, truth, s1_eval)
    s1 = read_source(data / "train" / "train_source1.tsv").select("entity_id", "country") \
        .filter(_eval_s1(pl.col("entity_id")))
    per_country = {c: macro_f05(decide(ev, decision), truth, s1.filter(pl.col("country") == c)["entity_id"])
                   for c in s1["country"].unique().sort().to_list()}
    imp = sorted(zip(feats, booster.feature_importance("gain")), key=lambda r: -r[1])
    with open(work / f"stage2{tag}_report.json", "w") as f:
        json.dump({**decision, "per_country": per_country, "stage1_oof": s1_decision,
                   "best_iteration": booster.best_iteration, "n_eval_s1": len(s1_eval),
                   "features": feats, "importance": [(a, float(b)) for a, b in imp]}, f, indent=1)
    _log(f"stage-2{tag} decision: {decision['rule']} f05={decision['metrics']['f05']:.5f} "
         f"(stage-1 OOF: {s1_decision['metrics']['f05']:.5f})")


def predict_stage2(data: Path, work: Path, out_dir: Path, tag: str = "") -> None:
    report = json.load(open(work / f"stage2{tag}_report.json"))
    booster = lgb.Booster(model_file=str(work / f"stage2{tag}.txt"))
    feats = booster.feature_name()
    preds = []
    for df in iter_stage2(work, "test"):
        preds.append(df.select(ID_COLS).with_columns(pl.Series("p", booster.predict(_X(df, feats), num_threads=config.N_THREADS))))
    pairs = pl.concat(preds)
    pairs.write_parquet(work / f"test_pred_s2{tag}.parquet")
    s1_ids = read_source(data / "test" / "test_source1.tsv")["entity_id"]
    matches = decide(pairs, report)
    write_id_lists(s1_ids, pairs.select("s1_id", pl.col("q_id").alias("rid")), "candidate_entity_ids",
                   out_dir / "candidate_pairs.tsv")
    write_id_lists(s1_ids, matches, "matched_entity_ids", out_dir / "matching_results.tsv")
    _log(f"wrote {out_dir}: {len(matches):,} matches for {matches['s1_id'].n_unique():,} of {len(s1_ids):,} S1 entities")


def step_stage2(data: Path, work: Path, out_dir: Path) -> None:
    for split in ("train", "test"):
        if not (work / split / "extra").exists():
            enrich(work, split)
    fit_stage1_folds(data, work)
    for split in ("train", "test"):
        predict_stage1(work, split)
        build_stage2_features(work, split)
    fit_stage2(data, work)
    predict_stage2(data, work, out_dir)
    if config.SELF_TRAIN:
        feats = lgb.Booster(model_file=str(work / "stage2.txt")).feature_name()
        fit_stage2(data, work, extra=pseudo_rows(work, feats), tag="_selftrain")
        predict_stage2(data, work, out_dir / "selftrain", tag="_selftrain")

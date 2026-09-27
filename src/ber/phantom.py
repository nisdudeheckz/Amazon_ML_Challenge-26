"""Orphan-record ("phantom S1") experiment: train and tune stage 2 on a test-like split.

Test has ~5.5-5.8 S2/S3 records per S1 entity in every country, train 4.68, while the
number of true matches per S1 looks the same (~3.4-3.5). The simplest explanation: the
test S1 set is a subsample and the records of the dropped S1 entities stayed in as
*orphans* - records with no true S1 at all. An orphan's best (wrong) candidate then
wins the record-level competition unopposed, a situation stage 2 never sees in train.

Simulation on train: a hash-selected PHANTOM per mille of S1 entities is removed
(180 -> 4.68 / 0.82 = 5.7 records per S1, as in test). Their records stay, and all
features that depend on the record's candidate list (rank, top1/top2, gaps, S1-side
context) are recomputed without them. Then, with the stage-1 models of the normal run:

  1. root stage-2 model + its decision rule, scored on the phantom held-out entities
     -> how much the orphan effect costs (the hypothesis test);
  2. the same predictions with the decision rule re-tuned on the phantom held-out;
  3. a stage-2 model refitted on the phantom training rows, tuned on the phantom
     held-out, and also checked on the normal held-out.

Test predictions of (2) and (3) are written to <out>/rootph/ and <out>/ph/.
Report: work/phantom_report.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from . import collective, config
from .candidates import KEEP_RATIO
from .io import read_source, write_id_lists
from .metrics import macro_f05
from .model import ID_COLS, PARAMS, _eval_s1, _train_q, choose_decision, decide
from .stage2 import S2_ROUNDS, _X, _fold, _labels, _parts, iter_stage2, stage2_aggregates

COLL = collective.COLL if config.COLLECTIVE else ()
TAG = "ph2" if config.COLLECTIVE else "ph"          # model / output name of the refitted stage 2
REPORT = "phantom2_report" if config.COLLECTIVE else "phantom_report"
LIST_COLS = ("bscore", "brank", "top1", "top2", "n_ret", "gap_top1", "top_margin", "ratio_top1")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_phantom(expr: pl.Expr) -> pl.Expr:
    return (expr.hash(41) % 1000) < config.PHANTOM


def phantom_view(df: pl.DataFrame) -> pl.DataFrame:
    """Drop phantom-S1 pairs and recompute the candidate-list features of each record."""
    schema = df.schema
    d = df.filter(~is_phantom(pl.col("s1_id")))
    d = d.with_columns(
        pl.col("bscore").max().over("q_id").alias("_t1"),
        pl.col("bscore").sort(descending=True).get(1, null_on_oob=True).over("q_id").alias("_t2"),
        pl.col("bscore").rank("ordinal", descending=True).over("q_id").alias("_rk"),
    ).with_columns(
        # a lone survivor: the true second score was never kept; if the old top1 is gone
        # it is below KEEP_RATIO * old top1, otherwise the old top2 still holds
        pl.when(pl.col("_t2").is_not_null()).then(pl.col("_t2"))
        .when(pl.col("_t1") == pl.col("top1")).then(pl.col("top2"))
        .otherwise(pl.min_horizontal(pl.col("_t1"), KEEP_RATIO * pl.col("top1"))).alias("_t2"),
    ).with_columns(
        (pl.col("_rk") - 1).alias("brank"), pl.col("_t1").alias("top1"), pl.col("_t2").alias("top2"),
    ).with_columns(
        (pl.col("bscore") - pl.col("top1")).alias("gap_top1"),
        (pl.col("top1") - pl.col("top2")).alias("top_margin"),
        (pl.col("bscore") / pl.col("top1")).alias("ratio_top1"),
    ).drop("_t1", "_t2", "_rk")
    return d.with_columns([pl.col(c).cast(schema[c]) for c in LIST_COLS if c in schema])


def phantom_context(frames: list[pl.DataFrame]) -> pl.DataFrame:
    """candidates.s1_context on the phantom view."""
    lf = pl.concat(frames).lazy()
    return lf.with_columns(
        pl.len().over("s1_id").cast(pl.Int32).alias("s1_nq"),
        (pl.col("brank") == 0).sum().over("s1_id").cast(pl.Int32).alias("s1_ntop1"),
        pl.col("bscore").max().over("s1_id").alias("s1_best"),
        pl.col("bscore").rank("ordinal", descending=True).over("s1_id").cast(pl.Int32).alias("s1_rank"),
    ).with_columns((pl.col("bscore") - pl.col("s1_best")).alias("s1_gap")) \
        .select("q_id", "s1_id", "s1_nq", "s1_ntop1", "s1_rank", "s1_gap").collect()


def _part(work: Path, p: Path) -> pl.DataFrame:
    df = phantom_view(pl.read_parquet(p))
    extra = work / "train" / "extra" / p.name
    if extra.exists():
        df = df.join(pl.read_parquet(extra), on=ID_COLS, how="left")
    return df


def _metrics(ev: pl.DataFrame, report: dict, truth: pl.DataFrame, s1: pl.DataFrame) -> dict:
    pred = decide(ev, report)
    out = {"all": macro_f05(pred, truth, s1["entity_id"])}
    for c in s1["country"].unique().sort().to_list():
        out[c] = macro_f05(pred, truth, s1.filter(pl.col("country") == c)["entity_id"])
    return out


def _brief(m: dict) -> str:
    return f"f05={m['f05']:.5f} single={m['f05_singletons']:.4f} P={m['pair_precision']:.4f} R={m['pair_recall']:.4f}"


def run(data: Path, work: Path, out_dir: Path) -> dict:
    truth, labels = _labels(data)
    parts = _parts(work, "train")
    s1_all = read_source(data / "train" / "train_source1.tsv").select("entity_id", "country")
    s1_eval_norm = s1_all.filter(_eval_s1(pl.col("entity_id")))
    s1_eval_ph = s1_eval_norm.filter(~is_phantom(pl.col("entity_id")))
    _log(f"phantom {config.PHANTOM}/1000: {s1_all.filter(is_phantom(pl.col('entity_id'))).height:,} of "
         f"{s1_all.height:,} train S1 removed; {s1_eval_ph.height:,} held-out S1 remain")

    # stage 1 in the phantom world (normal-run fold models, list features recomputed)
    ctx = phantom_context([phantom_view(pl.read_parquet(p, columns=[*ID_COLS, *LIST_COLS])).select(*ID_COLS, "bscore", "brank")
                           for p in parts])
    models = [lgb.Booster(model_file=str(work / f"stage1_fold{f}.txt")) for f in (0, 1)]
    f1 = models[0].feature_name()
    p1s = []
    for p in parts:
        df = _part(work, p).join(ctx, on=ID_COLS, how="left").with_columns(_fold(pl.col("q_id")).alias("_f"))
        p1 = np.zeros(len(df), np.float32)
        for f in (0, 1):
            m = (df["_f"] == f).to_numpy()
            if m.any():
                p1[m] = models[1 - f].predict(_X(df.filter(pl.col("_f") == f), f1), num_threads=config.N_THREADS)
        p1s.append(df.select(ID_COLS).with_columns(pl.Series("p1", p1), pl.lit(p.stem).alias("part")))
    del models
    p1 = pl.concat(p1s)
    del p1s
    eval_q = p1.filter(_eval_s1(pl.col("s1_id"))).select("q_id").unique()
    s2 = stage2_aggregates(p1, ctx)
    del ctx
    if COLL:
        qtok, s1tok = collective.record_tokens(work, "train")
        s2 = s2.join(collective.collective_features(p1, qtok, s1tok), on=ID_COLS, how="left")
        del qtok, s1tok
        _log("collective features (phantom world) built")
    s2 = s2.partition_by("part", as_dict=True, include_key=False)
    del p1
    _log("phantom stage-1 scores and stage-2 features built")

    # stage-2 rows in the phantom world
    root = lgb.Booster(model_file=str(work / "stage2.txt"))
    root_feats = root.feature_name()
    feats = root_feats + list(COLL)     # root features first: root scores Xev[:, :len(root_feats)]
    Xtr, ytr, Xes, yes, ev_ids, Xev = [], [], [], [], [], []
    for p in parts:
        if (p.stem,) not in s2:
            continue
        df = _part(work, p).join(s2[(p.stem,)], on=ID_COLS, how="left")
        tr = df.join(eval_q, on="q_id", how="anti").filter(_train_q(pl.col("q_id"))) \
            .join(labels, on=ID_COLS, how="left").with_columns(pl.col("y").fill_null(0))
        ev = df.join(eval_q, on="q_id", how="semi").join(labels, on=ID_COLS, how="left").with_columns(pl.col("y").fill_null(0))
        es = ev.filter((pl.col("q_id").hash(23) % 10) < 3)
        Xtr.append(_X(tr, feats)); ytr.append(tr["y"].to_numpy())
        Xes.append(_X(es, feats)); yes.append(es["y"].to_numpy())
        Xev.append(_X(ev, feats)); ev_ids.append(ev.select(ID_COLS))
        del df, tr, ev, es
    del s2
    Xev = np.concatenate(Xev)
    ev_ids = pl.concat(ev_ids)

    report = {"phantom_per_mille": config.PHANTOM, "n_eval_s1_phantom": s1_eval_ph.height}
    root_report = json.load(open(work / "stage2_report.json"))

    # 1. root model + root decision, phantom world
    ev_root = ev_ids.with_columns(pl.Series("p", root.predict(Xev[:, :len(root_feats)], num_threads=config.N_THREADS)))
    report["root_on_phantom"] = _metrics(ev_root, root_report, truth, s1_eval_ph)
    report["root_on_normal"] = root_report["metrics"]
    _log(f"ROOT normal : {_brief(root_report['metrics'])}")
    _log(f"ROOT phantom: {_brief(report['root_on_phantom']['all'])}")
    # 2. root model, decision re-tuned in the phantom world
    dec_rootph = choose_decision(ev_root, truth, s1_eval_ph["entity_id"])
    report["rootph_decision"] = {k: dec_rootph[k] for k in ("rule", "lam", "floor", "threshold")}
    report["rootph_on_phantom"] = _metrics(ev_root, dec_rootph, truth, s1_eval_ph)
    ev_root_norm = pl.read_parquet(work / "eval_pred_s2.parquet")   # root model, normal held-out
    report["rootph_on_normal"] = _metrics(ev_root_norm, dec_rootph, truth, s1_eval_norm)
    _log(f"ROOT+phantom-tuned decision {report['rootph_decision']}: phantom {_brief(report['rootph_on_phantom']['all'])} "
         f"normal {_brief(report['rootph_on_normal']['all'])}")
    del root, ev_root_norm

    # 3. stage 2 refitted on phantom rows
    ytr, yes = np.concatenate(ytr), np.concatenate(yes)
    _log(f"stage-2 phantom fit pairs: {len(ytr):,} ({int(ytr.sum()):,} positive)")
    dtr = lgb.Dataset(Xtr, ytr, feature_name=feats, free_raw_data=True)
    dev = lgb.Dataset(np.concatenate(Xes), yes, reference=dtr)
    del Xtr, Xes
    booster = lgb.train(PARAMS, dtr, S2_ROUNDS, valid_sets=[dev], valid_names=["eval"],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    del dtr, dev
    booster.save_model(str(work / f"stage2_{TAG}.txt"))
    ev_ph = ev_ids.with_columns(pl.Series("p", booster.predict(Xev, num_threads=config.N_THREADS)))
    del Xev
    dec_ph = choose_decision(ev_ph, truth, s1_eval_ph["entity_id"])
    report["ph_decision"] = {k: dec_ph[k] for k in ("rule", "lam", "floor", "threshold")}
    report["ph_on_phantom"] = _metrics(ev_ph, dec_ph, truth, s1_eval_ph)
    # ... and in the normal world (must not lose much there)
    norm_q = pl.scan_parquet(work / "train" / "p1.parquet").filter(_eval_s1(pl.col("s1_id"))).select("q_id").unique().collect()
    cf = _normal_collective(work, "train")
    preds = []
    for df in iter_stage2(work, "train"):
        df = df.join(norm_q, on="q_id", how="semi")
        if cf is not None:
            df = df.join(cf, on=ID_COLS, how="left")
        preds.append(df.select(ID_COLS).with_columns(pl.Series("p", booster.predict(_X(df, feats), num_threads=config.N_THREADS))))
    ev_norm = pl.concat(preds)
    report["ph_on_normal"] = _metrics(ev_norm, dec_ph, truth, s1_eval_norm)
    report["ph_on_normal_retuned"] = choose_decision(ev_norm, truth, s1_eval_norm["entity_id"])["metrics"]
    _log(f"PH  phantom : {_brief(report['ph_on_phantom']['all'])}  decision {report['ph_decision']}")
    _log(f"PH  normal  : {_brief(report['ph_on_normal']['all'])} (re-tuned {_brief(report['ph_on_normal_retuned'])})")
    json.dump({**dec_ph, "best_iteration": booster.best_iteration, "features": feats},
              open(work / f"stage2_{TAG}_report.json", "w"), indent=1)
    with open(work / f"{REPORT}.json", "w") as f:
        json.dump(report, f, indent=1)

    # test outputs: root predictions with the phantom-tuned rule, and the phantom model
    s1_test = read_source(data / "test" / "test_source1.tsv")["entity_id"]
    if not COLL:
        tp = pl.read_parquet(work / "test_pred_s2.parquet")
        _write(tp, dec_rootph, s1_test, out_dir / "rootph")
    cf = _normal_collective(work, "test")
    tp = []
    for df in iter_stage2(work, "test"):
        if cf is not None:
            df = df.join(cf, on=ID_COLS, how="left")
        tp.append(df.select(ID_COLS).with_columns(pl.Series("p", booster.predict(_X(df, feats), num_threads=config.N_THREADS))))
    tp = pl.concat(tp)
    tp.write_parquet(work / f"test_pred_s2_{TAG}.parquet")
    _write(tp, dec_ph, s1_test, out_dir / TAG)
    return report


def _normal_collective(work: Path, split: str) -> pl.DataFrame | None:
    """Collective features of the normal (unmodified) candidate set of a split."""
    if not COLL:
        return None
    qtok, s1tok = collective.record_tokens(work, split)
    cf = collective.collective_features(pl.read_parquet(work / split / "p1.parquet", columns=[*ID_COLS, "p1"]), qtok, s1tok)
    _log(f"collective features ({split}, normal) built: {cf.shape}")
    return cf


def _write(pairs: pl.DataFrame, decision: dict, s1_ids: pl.Series, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    matches = decide(pairs, decision)
    write_id_lists(s1_ids, pairs.select("s1_id", pl.col("q_id").alias("rid")), "candidate_entity_ids",
                   out / "candidate_pairs.tsv")
    write_id_lists(s1_ids, matches, "matched_entity_ids", out / "matching_results.tsv")
    _log(f"wrote {out}: {len(matches):,} matches for {matches['s1_id'].n_unique():,} of {len(s1_ids):,} S1 entities")

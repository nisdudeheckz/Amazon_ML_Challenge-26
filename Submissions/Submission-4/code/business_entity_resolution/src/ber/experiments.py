"""Cross-country transfer experiments (a proxy for the unseen test country).

France only appears in the test set, so its score cannot be measured offline. The
closest proxy: train on one training country, tune the decision rule on that country's
held-out entities, then score the *other* country — exactly how the pipeline treats
France. Comparing feature groups this way shows which features transfer and which
only fit the training countries.

Output: work/xval_report.json (and a table in the log).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from .io import read_ground_truth, read_source
from .metrics import macro_f05
from .model import ID_COLS, PARAMS, _eval_s1, choose_decision, decide, ensure_context

ROUNDS = 600
FIT_FRAC = 300      # per mille of the source country's records used for fitting
TARGET_FRAC = 200   # per mille of the target country's S1 entities that are scored


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _target_s1(expr: pl.Expr) -> pl.Expr:
    return (expr.hash(29) % 1000) < TARGET_FRAC


def _country_parts(work: Path, country: str) -> list[Path]:
    return sorted((work / "train" / "parts").glob(f"{country}_*.parquet"))


def _load(work: Path, p: Path, ctx: pl.DataFrame) -> pl.DataFrame:
    df = pl.read_parquet(p).join(ctx, on=ID_COLS, how="left")
    extra = work / "train" / "extra" / p.name
    if extra.exists():
        df = df.join(pl.read_parquet(extra), on=ID_COLS, how="left")
    return df


def feature_groups(all_feats: list[str]) -> dict[str, list[str]]:
    ratio = [f for f in all_feats if f.startswith(("nm_qx_", "nm_sx_"))
             and f.endswith(("_noise", "_vocab", "_maxlratio", "_typo", "_n", "_minjw"))]
    name_abs = [f for f in all_feats if f.startswith(("nm_qx_", "nm_sx_")) and f not in ratio]
    legal_code = [f for f in all_feats if f.startswith(("legal_q_only", "legal_s_only", "code_"))
                  or f == "legal_equal"]
    base = [f for f in all_feats if f not in ratio + name_abs + legal_code]
    return {"base": base, "base+legal_code": base + legal_code,
            "base+legal_code+ratio": base + legal_code + ratio, "full": all_feats}


def _queries_touching(parts: list[Path], mask) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(p, columns=ID_COLS).filter(mask(pl.col("s1_id"))).select("q_id")
                      for p in parts]).unique()


def cross_country(data: Path, work: Path) -> dict:
    truth = read_ground_truth(data / "train" / "train_ground_truth.tsv").rename({"match_id": "rid"})
    labels = truth.select("s1_id", pl.col("rid").alias("q_id"), pl.lit(1, pl.Int8).alias("y"))
    ctx = pl.read_parquet(ensure_context(work, "train"))
    s1 = read_source(data / "train" / "train_source1.tsv").select("entity_id", "country")
    countries = sorted({p.name.rsplit("_", 1)[0] for p in (work / "train" / "parts").glob("*.parquet")})
    feats_all = [c for c in _load(work, _country_parts(work, countries[0])[0], ctx).columns if c not in ID_COLS]
    groups = feature_groups(feats_all)
    results = []
    for src in countries:
        src_parts = _country_parts(work, src)
        heldout_q = _queries_touching(src_parts, _eval_s1)
        fit, held = [], []
        for p in src_parts:
            df = _load(work, p, ctx).join(labels, on=ID_COLS, how="left").with_columns(pl.col("y").fill_null(0))
            fit.append(df.join(heldout_q, on="q_id", how="anti").filter((pl.col("q_id").hash(13) % 1000) < FIT_FRAC))
            held.append(df.join(heldout_q, on="q_id", how="semi"))
        fit, held = pl.concat(fit), pl.concat(held)
        src_eval_ids = s1.filter((pl.col("country") == src) & _eval_s1(pl.col("entity_id")))["entity_id"]
        for dst in countries:
            if dst == src:
                continue
            dst_parts = _country_parts(work, dst)
            dst_q = _queries_touching(dst_parts, _target_s1)
            dst_df = pl.concat([_load(work, p, ctx).join(dst_q, on="q_id", how="semi") for p in dst_parts])
            dst_ids = s1.filter((pl.col("country") == dst) & _target_s1(pl.col("entity_id")))["entity_id"]
            for gname, feats in groups.items():
                t = time.time()
                booster = lgb.train(PARAMS, lgb.Dataset(fit.select(pl.col(feats).cast(pl.Float32)).to_numpy(),
                                                        fit["y"].to_numpy(), feature_name=feats), ROUNDS)
                pred = lambda df: df.select(ID_COLS).with_columns(pl.Series(
                    "p", booster.predict(df.select(pl.col(feats).cast(pl.Float32)).to_numpy())))
                decision = choose_decision(pred(held), truth, src_eval_ids)       # tuned on source only
                same = decision["metrics"]
                cross = macro_f05(decide(pred(dst_df), decision), truth, dst_ids)
                row = {"train": src, "eval": dst, "features": gname, "n_features": len(feats),
                       "same_country_f05": same["f05"], "cross_country_f05": cross["f05"],
                       "cross_precision": cross["pair_precision"], "cross_recall": cross["pair_recall"],
                       "rule": decision["rule"], "minutes": round((time.time() - t) / 60, 1)}
                results.append(row)
                _log(f"XVAL {src}->{dst} {gname:16s} same={same['f05']:.5f} cross={cross['f05']:.5f} "
                     f"P={cross['pair_precision']:.4f} R={cross['pair_recall']:.4f}")
    report = {"rounds": ROUNDS, "target_frac": TARGET_FRAC / 1000, "results": results}
    with open(work / "xval_report.json", "w") as f:
        json.dump(report, f, indent=1)
    return report

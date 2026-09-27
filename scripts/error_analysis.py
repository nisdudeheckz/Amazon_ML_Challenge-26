"""Inspect held-out errors: python scripts/error_analysis.py [--work-dir work] [--n 15]

Uses work/eval_pred.parquet + work/train_report.json written by `ber.run train`.
Prints per-country metrics, sample false positives / false negatives, and blocking
misses for the held-out S1 entities.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ber.io import read_ground_truth, read_source  # noqa: E402
from ber.metrics import macro_f05  # noqa: E402
from ber.model import _eval_s1, decide  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, default=ROOT / "work")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "student_resource" / "dataset")
    ap.add_argument("--n", type=int, default=15)
    args = ap.parse_args()
    report = json.load(open(args.work_dir / "train_report.json"))
    ev = pl.read_parquet(args.work_dir / "eval_pred.parquet")
    truth = read_ground_truth(args.data_dir / "train" / "train_ground_truth.tsv").rename({"match_id": "rid"})
    s1 = read_source(args.data_dir / "train" / "train_source1.tsv")
    rec = pl.concat([s1, read_source(args.data_dir / "train" / "train_source2.tsv"),
                     read_source(args.data_dir / "train" / "train_source3.tsv")])
    s1e = s1.filter(_eval_s1(pl.col("entity_id")))
    pred = decide(ev, report)
    for c in s1e["country"].unique().sort().to_list():
        ids = s1e.filter(pl.col("country") == c)["entity_id"]
        print(c, macro_f05(pred, truth, ids))
    t = truth.filter(pl.col("s1_id").is_in(s1e["entity_id"].implode()))
    p = pred.filter(pl.col("s1_id").is_in(s1e["entity_id"].implode()))
    fp = p.join(t, on=["s1_id", "rid"], how="anti")
    fn = t.join(p, on=["s1_id", "rid"], how="anti")
    cand = ev.select(pl.col("s1_id"), pl.col("q_id").alias("rid"), "p")
    fn_blk = fn.join(cand, on=["s1_id", "rid"], how="anti")
    print(f"FP pairs {len(fp):,}  FN pairs {len(fn):,} (of which not in candidates: {len(fn_blk):,})")
    look = rec.select("entity_id", "business_name", "business_address")
    d = {r[0]: r[1:] for r in look.filter(pl.col("entity_id").is_in(
        pl.concat([fp["s1_id"], fp["rid"], fn["s1_id"], fn["rid"]]).implode())).iter_rows()}
    true_of = dict(truth.select("rid", "s1_id").iter_rows())
    for title, df in (("FALSE POSITIVES", fp), ("FALSE NEGATIVES (in candidates)", fn.join(fn_blk, on=["s1_id", "rid"], how="anti")),
                      ("FALSE NEGATIVES (blocking miss)", fn_blk)):
        print(f"\n===== {title}")
        for s, r in df.sample(min(args.n, len(df)), seed=0).iter_rows():
            pp = cand.filter((pl.col("s1_id") == s) & (pl.col("rid") == r))["p"]
            print(f"S1 {d.get(s)}\n   {r} {d.get(r)}  p={pp[0] if len(pp) else None:}  true_s1={true_of.get(r)}")


if __name__ == "__main__":
    main()

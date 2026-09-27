import json, time, sys
import numpy as np, polars as pl, lightgbm as lgb
sys.path.insert(0, "code_root/src")
from pathlib import Path
from ber.stage2 import iter_stage2, _eval_queries, _labels, _X
from ber.model import _eval_s1, choose_decision, decide
from ber.metrics import macro_f05
from ber.io import read_source, write_id_lists
W = Path("work"); t0 = time.time()
truth, _ = _labels(Path("student_resource/dataset"))
eval_q = _eval_queries(W)
A = lgb.Booster(model_file=str(W / "stage2.txt")); fa = A.feature_name()
B = lgb.Booster(model_file=str(W / "stage2_s5like.txt")); fb = B.feature_name()
ev = []
for df in iter_stage2(W, "train"):
    d = df.join(eval_q, on="q_id", how="semi")
    ev.append(d.select("q_id", "s1_id").with_columns(pl.Series("pa", A.predict(_X(d, fa), num_threads=12)), pl.Series("pb", B.predict(_X(d, fb), num_threads=12))))
ev = pl.concat(ev); print(f"eval preds {time.time()-t0:.0f}s", flush=True)
s1 = read_source(Path("student_resource/dataset/train/train_source1.tsv")).select("entity_id", "country").filter(_eval_s1(pl.col("entity_id")))
sid = s1["entity_id"]
avg = ev.select("q_id", "s1_id", ((pl.col("pa") + pl.col("pb")) / 2).alias("p"))
dec = choose_decision(avg, truth, sid)
m = dec["metrics"]; per = {c: round(macro_f05(decide(avg, dec), truth, s1.filter(pl.col("country") == c)["entity_id"])["f05"], 5) for c in ("India", "US")}
print(f"AVG f05={m['f05']:.5f} single={m['f05_singletons']:.4f} P={m['pair_precision']:.4f} R={m['pair_recall']:.4f} {per} rule={dec['rule']} lam={dec['lam']} floor={dec['floor']} t={dec['threshold']}", flush=True)
json.dump({"metrics": m, "rule": dec["rule"], "lam": dec["lam"], "floor": dec["floor"], "threshold": float(dec["threshold"])}, open(W / "s8_report.json", "w"), indent=1)
tp = []
for df in iter_stage2(W, "test"):
    tp.append(df.select("q_id", "s1_id").with_columns(pl.Series("pb", B.predict(_X(df, fb), num_threads=12))))
tp = pl.concat(tp).join(pl.read_parquet(W / "test_pred_s2.parquet"), on=["q_id", "s1_id"], how="left")
tp = tp.select("q_id", "s1_id", ((pl.col("p") + pl.col("pb")) / 2).alias("p"))
out = Path("output-s8"); out.mkdir(exist_ok=True)
s1t = read_source(Path("student_resource/dataset/test/test_source1.tsv"))["entity_id"]
mt = decide(tp, dec)
write_id_lists(s1t, tp.select("s1_id", pl.col("q_id").alias("rid")), "candidate_entity_ids", out / "candidate_pairs.tsv")
write_id_lists(s1t, mt, "matched_entity_ids", out / "matching_results.tsv")
print(f"wrote S8: {len(mt):,} matches, {time.time()-t0:.0f}s", flush=True)

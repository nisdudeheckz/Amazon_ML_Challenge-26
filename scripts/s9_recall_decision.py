import json, sys, polars as pl
sys.path.insert(0, "code_root/src")
from pathlib import Path
from ber.model import _eval_s1, assign_expected_f
from ber.metrics import macro_f05
from ber.io import read_source, read_ground_truth, write_id_lists
truth = read_ground_truth(Path("student_resource/dataset/train/train_ground_truth.tsv")).rename({"match_id": "rid"})
sid = read_source(Path("student_resource/dataset/train/train_source1.tsv")).select("entity_id").filter(_eval_s1(pl.col("entity_id")))["entity_id"]
ev = pl.read_parquet("work/eval_pred_s2.parquet")
best = None
for lam, floor in [(0.0, 0.4), (0.5, 0.4), (0.5, 0.3), (1.0, 0.3), (1.0, 0.2), (2.0, 0.2)]:
    m = macro_f05(assign_expected_f(ev, lam, floor), truth, sid)
    print(f"lam={lam} floor={floor} f05={m['f05']:.5f} single={m['f05_singletons']:.4f} P={m['pair_precision']:.4f} R={m['pair_recall']:.4f}", flush=True)
    if m["f05"] >= 0.98840 and (best is None or m["pair_recall"] > best[2]["pair_recall"]):
        best = (lam, floor, m)
lam, floor, m = best
print("CHOSEN", lam, floor, round(m["f05"], 5), round(m["pair_recall"], 4), flush=True)
tp = pl.read_parquet("work/test_pred_s2.parquet")
out = Path("output-s9"); out.mkdir(exist_ok=True)
s1t = read_source(Path("student_resource/dataset/test/test_source1.tsv"))["entity_id"]
mt = assign_expected_f(tp, lam, floor)
write_id_lists(s1t, tp.select("s1_id", pl.col("q_id").alias("rid")), "candidate_entity_ids", out / "candidate_pairs.tsv")
write_id_lists(s1t, mt, "matched_entity_ids", out / "matching_results.tsv")
json.dump({"rule": "expected_f", "lam": lam, "floor": floor, "metrics": m}, open("work/s9_report.json", "w"), indent=1)
print(f"wrote S9: {len(mt):,} matches", flush=True)

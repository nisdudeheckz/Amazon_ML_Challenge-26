import json, time, sys
import numpy as np, polars as pl, lightgbm as lgb
sys.path.insert(0, "code_root/src")
from pathlib import Path
from ber import config
from ber.stage2 import iter_stage2, _eval_queries, _labels, _feats, _X, S2_ROUNDS
from ber.model import PARAMS, _train_q, _eval_s1, choose_decision, decide
from ber.metrics import macro_f05
from ber.io import read_source
W = Path("work"); t0 = time.time()
NEW = {"s_addr_n", "s_addr_w1_n", "s_support", "q_addr_s1n", "q_support", "nm_qx_maxpct", "nm_qx_minpct", "nm_sx_maxpct", "nm_sx_minpct"}
truth, labels = _labels(Path("student_resource/dataset"))
eval_q = _eval_queries(W)
root = lgb.Booster(model_file=str(W / "stage2.txt")); rf = root.feature_name()
feats = [f for f in rf if f not in NEW] + ["q_name_freq", "s_name_freq"]   # S5-like feature set
Xtr, ytr, Xes, yes, Xev, ids = [], [], [], [], [], []
for df in iter_stage2(W, "train"):
    tr = df.join(eval_q, on="q_id", how="anti").filter(_train_q(pl.col("q_id"))).join(labels, on=["q_id","s1_id"], how="left").with_columns(pl.col("y").fill_null(0))
    ev = df.join(eval_q, on="q_id", how="semi").join(labels, on=["q_id","s1_id"], how="left").with_columns(pl.col("y").fill_null(0))
    es = ev.filter((pl.col("q_id").hash(23) % 10) < 3)
    Xtr.append(_X(tr, feats)); ytr.append(tr["y"].to_numpy()); Xes.append(_X(es, feats)); yes.append(es["y"].to_numpy())
    Xev.append(_X(ev, feats)); ids.append(ev.select("q_id", "s1_id"))
print(f"rows built {time.time()-t0:.0f}s", flush=True)
ytr = np.concatenate(ytr); dtr = lgb.Dataset(Xtr, ytr, feature_name=feats, free_raw_data=True)
dev = lgb.Dataset(np.concatenate(Xes), np.concatenate(yes), reference=dtr); del Xtr, Xes
b = lgb.train({**PARAMS, "num_threads": 12}, dtr, S2_ROUNDS, valid_sets=[dev], callbacks=[lgb.early_stopping(50, verbose=False)])
print(f"fit {time.time()-t0:.0f}s best_it {b.best_iteration}", flush=True)
ids = pl.concat(ids); evB = ids.with_columns(pl.Series("p", b.predict(np.concatenate(Xev), num_threads=12)))
s1 = read_source(Path("student_resource/dataset/train/train_source1.tsv")).select("entity_id", "country").filter(_eval_s1(pl.col("entity_id")))
sid = s1["entity_id"]
evA = pl.read_parquet(W / "eval_pred_s2.parquet")
repA = json.load(open(W / "stage2_report.json"))
decB = choose_decision(evB, truth, sid)
A = decide(evA, repA).unique(); B = decide(evB, decB).unique()
def union(first, second):   # one S1 per record: `first` wins conflicts
    return pl.concat([first, second.join(first, on="rid", how="anti")]).unique()
inter = A.join(B, on=["s1_id", "rid"], how="semi")
res = {}
for name, pr in [("A_root(S7-like)", A), ("B_S5-like", B), ("union_A_first", union(A, B)), ("union_B_first", union(B, A)), ("intersection", inter)]:
    m = macro_f05(pr, truth, sid)
    per = {c: round(macro_f05(pr, truth, s1.filter(pl.col("country") == c)["entity_id"])["f05"], 5) for c in ("India", "US")}
    res[name] = m
    print(f"{name:16s} f05={m['f05']:.5f} single={m['f05_singletons']:.4f} P={m['pair_precision']:.4f} R={m['pair_recall']:.4f} pairs={pr.height:,} {per}", flush=True)
print(f"changed vs A: union adds {union(A,B).height - A.height:,} pairs; total {time.time()-t0:.0f}s")
b.save_model(str(W / "stage2_s5like.txt")); json.dump({k: v for k, v in decB.items() if k in ("rule","lam","floor","threshold")}, open(W / "s5like_decision.json", "w"), default=float)

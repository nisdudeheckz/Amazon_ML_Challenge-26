# Business Entity Resolution — reproducible pipeline

Blocking (TF-IDF top-K retrieval) → pair features → LightGBM pair classifier →
per-record assignment + F0.5-aware decision → `matching_results.tsv` / `candidate_pairs.tsv`.

Only the provided training/test files are used. No external data, APIs or pretrained
models; the only "model" is a LightGBM (MIT) gradient-boosted tree ensemble trained from
scratch (≈ a few MB, far below the 8B-parameter cap).

## Environment

Tested with Python 3.11.9 on Windows 11 (16 GB RAM, 16 logical CPUs). No GPU needed.

```bash
python -m venv .venv
.venv/Scripts/activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce end-to-end

`DATA` is the challenge `dataset/` folder containing `train/` and `test/`.

```bash
cd src
python -m ber.run all --data-dir DATA --work-dir ../work --out-dir ../output
```

This runs, in order (each step can also be run on its own):

| step | what it does | output |
|---|---|---|
| `prepare` | learns native-script→Latin token maps from train ground truth; normalises all 6 source files | `work/translit.json`, `work/norm/*.parquet` |
| `candidates` | per country: TF-IDF index of S1, top-20 retrieval for every S2/S3 record, pruning, pair features | `work/{train,test}/parts/*.parquet` |
| `train` | LightGBM on train pairs, held-out-S1 validation, decision-rule tuning, refit | `work/model_final.txt`, `work/train_report.json` |
| `predict` | scores all test candidate pairs and writes the two submission files | `output/matching_results.tsv`, `output/candidate_pairs.tsv` |

Runtime on the reference machine: ~1.5–2 h end to end (candidate generation dominates).
Peak memory ≈ 12 GB. `--splits test` restricts `candidates` to one split.

## Code layout

```
src/ber/
  io.py          TSV reading (explicit tab separator, no quoting) and submission writers
  translit.py    learns / applies native-script -> Latin token dictionaries (train only)
  normalize.py   vectorised name & address normalisation (polars)
  vectors.py     hashed TF-IDF namespaces, per-namespace CSR, numba pair-cosine kernels
  blocking.py    S1 index + multi-threaded sparse top-K retrieval (sparse_dot_topn)
  pairfeat.py    pair features (rapidfuzz cpdist, sparse cosines, numbers, legal forms, ...)
  candidates.py  per-country / per-chunk orchestration of blocking + features
  model.py       LightGBM training, validation protocol, decision rules, prediction
  metrics.py     macro F0.5 exactly as defined by the challenge
  run.py         CLI entry point
```

## Validate the output

```bash
python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

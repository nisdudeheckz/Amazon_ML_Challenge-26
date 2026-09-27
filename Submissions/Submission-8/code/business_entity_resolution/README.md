# Business Entity Resolution — reproducible pipeline

The pipeline runs in five stages:

1. **Blocking** — TF-IDF top-K retrieval.
2. **Pair features.**
3. **Stage-1 LightGBM** — two out-of-fold models.
4. **Stage-2 LightGBM** — adds competition features built from the stage-1 probabilities.
5. **Decision** — per-record assignment plus an F0.5-aware rule, producing `matching_results.tsv` and `candidate_pairs.tsv`.

Only the provided training and test files are used: no external data, APIs or pretrained models. The only models are LightGBM (MIT) gradient-boosted tree ensembles trained from scratch. They are a few MB each, far below the 8B-parameter cap.

## Environment

Tested with Python 3.11.9 on Windows 11 (16 GB RAM, 16 logical CPUs). No GPU is needed.

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

This runs the following steps in order. Each step can also be run on its own.

| step | what it does | output |
|---|---|---|
| `prepare` | Learns native-script→Latin token maps from the train ground truth; normalises all 6 source files. | `work/translit.json`, `work/norm/*.parquet` |
| `candidates` | Per country: TF-IDF index of S1 (idf over S1+S2+S3), top-20 retrieval for every S2/S3 record, pruning, pair features. | `work/{train,test}/parts/*.parquet` |
| `enrich` | Difference-type pair features: typo vs. vocabulary-word substitutions in names, one-sided legal forms, closest unit-code pair. `stage2` runs it automatically if its output is missing. | `work/{train,test}/extra/*.parquet` |
| `stage2` | Two stage-1 fold models → out-of-fold p1 on train and test → competition features → stage-2 LightGBM with held-out-S1 validation → decision-rule tuning → test output. | `work/stage2.txt`, `work/stage2_report.json`, `output/*.tsv` |

On the reference machine the full run takes about 4 hours; candidate generation dominates (≈3.5 h). Peak memory is about 12 GB.

The candidates step takes two options:

- `--splits test` restricts it to one split.
- `--countries India,US` rebuilds only some countries.

The single-stage variant (Submission-1) is `train` → `tune` → `predict`, run after `candidates`.

## Code layout

```
src/ber/
  io.py          TSV reading (explicit tab separator, no quoting) and submission writers
  translit.py    learns / applies native-script -> Latin token dictionaries (train only)
  normalize.py   vectorised name & address normalisation (polars)
  vectors.py     hashed TF-IDF namespaces, per-namespace CSR, numba pair-cosine kernels
  blocking.py    S1 index + multi-threaded sparse top-K retrieval (sparse_dot_topn)
  pairfeat.py    pair features (rapidfuzz cpdist, sparse cosines, house numbers, ...)
  enrich.py      difference-type features (typo vs substitution, legal-form changes, codes)
  candidates.py  per-country / per-chunk orchestration of blocking + features
  model.py       stage-1 training, validation protocol, decision rules
  stage2.py      out-of-fold stage-1 scoring, competition features, stage-2 model
  metrics.py     macro F0.5 exactly as defined by the challenge
  run.py         CLI entry point
```

## Validate the output

```bash
python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

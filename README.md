# Amazon ML Challenge 2026 — Business Entity Resolution

Given business records from three sources, the task is to find, for every **Source-1** entity, all **Source-2 / Source-3** records that refer to the same real-world business. Names and addresses are noisy: typos, transliterations, abbreviations, re-ordered and truncated addresses. Scoring is **macro F0.5 per S1 entity**. The training data covers the US and India; the test set adds France.

| submission | approach | held-out F0.5 | leaderboard |
|---|---|---|---|
| Submission-1 | TF-IDF blocking + LightGBM | 0.9689 | 0.9626 |
| Submission-2 | compound blocking + difference features + 2-stage LightGBM | **0.9885** | pending |

## How it works

```
normalise ──▶ blocking (top-K retrieval) ──▶ pair features ──▶ stage-1 LightGBM (2 folds, OOF)
                                                     │                       │
                                          enrich: difference-type      competition features
                                          features                           │
                                                     └──────────▶ stage-2 LightGBM ──▶ one-to-one
                                                                                     assignment +
                                                                                   F0.5 decision rule
```

1. **Normalisation** (`normalize.py`, `translit.py`)
   - Rule-based name and address cleaning: legal forms, honorifics, alias markers (`DBA`, `formerly`, …), leetspeak, street-type and unit-code canonicalisation.
   - A native-script → Latin token dictionary is *learned from training pairs*, covering Hindi, Gujarati, Telugu, Tamil, Kannada, Bengali, … names.
2. **Blocking** (`vectors.py`, `blocking.py`, `candidates.py`)
   - Every S2/S3 record belongs to at most one S1 entity, so retrieval runs from the S2/S3 side.
   - Each record becomes nine hashed TF-IDF namespaces: name words, name 3-grams, address words, numbers, unit codes, house-number × street, plus compound *name-pair*, *name × address* and *address-bigram* keys. The compound keys keep recall at **98.6%** while common tokens are pruned from the sparse product.
3. **Features** (`pairfeat.py`, `enrich.py`)
   - Similarity features: namespace cosines, rapidfuzz string similarities, house-number relations, retrieval and entity context.
   - *Difference-type* features describe what kind of change separates a pair, for example a typo vs. a word substituted by another real word, or a legal form that switched. Distractors in this data are mutations of real businesses, and this is what gives them away.
4. **Model** (`model.py`, `stage2.py`)
   - Stage 1 is a LightGBM trained twice, so every pair is scored out-of-fold.
   - Stage 2 adds competition features: this candidate vs. the record's other candidates, and vs. other records claiming the same entity.
   - Each record is assigned to at most one entity, using an expected-F0.5 decision rule tuned on held-out entities.

The full methodology write-up is in [docs/Documentation_template.md](docs/Documentation_template.md).

## Repository layout

```
src/ber/                     the pipeline (module guide: docs/CODE_README.md)
scripts/make_submission.py   packages Submissions/Submission-<n>/ (+ official validator, + zip)
scripts/error_analysis.py    held-out false positives / negatives
aws/                         run the pipeline on EC2 (see aws/README.md)
docs/                        methodology document + code README shipped in submissions
Submissions/Submission-<n>/  every submission in the required package layout
                             (code snapshot, docs, metrics; TSVs and zip are git-ignored)
requirements.txt             pinned dependencies (Python 3.11)
```

The organisers' `student_resource/` folder (dataset, validator, templates) is expected at the repo root but isn't tracked.

## Run locally

```bash
python -m venv .venv && .venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cd src
python -m ber.run all --data-dir ../student_resource/dataset --work-dir ../work --out-dir ../output
cd ..
python scripts/make_submission.py --team <TEAM_NAME>
```

A full run takes about 4 hours on a 16-thread, 16 GB laptop.

Runtime knobs are environment variables, listed in `src/ber/config.py`: `BER_THREADS`, `BER_JOBS` (countries in parallel), `BER_MAX_DF`, `BER_TRAIN_FRAC`.

## Run on AWS

```bash
aws/setup_once.sh          # bucket, instance role, dataset upload, quota check
aws/launch.sh full-1       # self-terminating EC2 run; the log is mirrored to S3
aws/status.sh full-1
aws/fetch.sh full-1        # outputs + exact code -> output-aws/full-1/
```

See [aws/README.md](aws/README.md) for costs and options, such as re-running only stage 2 from an earlier run's work directory.

## Rules followed

- Only the provided training and test data are used: no external lookups, APIs or geocoding.
- The models are LightGBM ensembles trained from scratch. They are MIT-licensed and a few MB each.
- All dependencies are MIT / BSD / Apache-2.0 / ISC.

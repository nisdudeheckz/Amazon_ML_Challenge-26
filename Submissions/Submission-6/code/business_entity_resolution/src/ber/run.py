"""End-to-end CLI.

    python -m ber.run all        --data-dir DATA --work-dir WORK --out-dir OUT
    python -m ber.run prepare    ...   # learn transliteration maps, normalise all sources
    python -m ber.run candidates ...   # blocking + pair features for train and test
    python -m ber.run enrich     ...   # difference-type pair features (also run by stage2 if missing)
    python -m ber.run stage2     ...   # 2-fold stage-1 models, stage-2 model, tuning, test output
    python -m ber.run xval       ...   # cross-country transfer experiments (proxy for unseen countries)
    python -m ber.run xval_st    ...   # the same with self-training on the target country
    python -m ber.run xval_ablate ...  # cross-country effect of dropping country-scale features
    python -m ber.run phantom    ...   # after stage2: stage 2 re-fit / re-tuned with orphan records

  single-stage variant (Submission-1):
    python -m ber.run train      ...   # LightGBM + threshold tuning on a held-out S1 split
    python -m ber.run tune       ...   # re-tune only the decision rule from saved predictions
    python -m ber.run predict    ...   # final model on test -> output/*.tsv

(run from the `src/` directory, or with `src/` on PYTHONPATH)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

from . import candidates, enrich, experiments, model, phantom, stage2, translit
from .io import read_ground_truth, read_source
from .normalize import normalize


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


NORM_SLICE = 1_000_000


def norm_paths(work: Path, split: str) -> dict[int, Path]:
    return {s: work / "norm" / f"{split}_s{s}.parquet" for s in (1, 2, 3)}


def step_prepare(data: Path, work: Path) -> None:
    """Resumable: an existing translit.json and finished norm files are reused (each norm
    file is written to a temporary name first, so a crash never leaves a partial one)."""
    (work / "norm").mkdir(parents=True, exist_ok=True)
    if (work / "translit.json").exists():
        maps = translit.load(work / "translit.json")
    else:
        raw = {s: read_source(data / "train" / f"train_source{s}.tsv") for s in (1, 2, 3)}
        pairs = read_ground_truth(data / "train" / "train_ground_truth.tsv")
        maps = translit.learn(raw[1], pl.concat([raw[2], raw[3]]), pairs)
        translit.save(maps, work / "translit.json")
        del raw, pairs
    _log(f"transliteration maps: {len(maps['name'])} name tokens, {len(maps['addr'])} address components")
    for split in ("train", "test"):
        for s in (1, 2, 3):
            out = norm_paths(work, split)[s]
            if out.exists():
                continue
            tmp = out.with_suffix(".tmp")
            raw = read_source(data / split / f"{split}_source{s}.tsv")
            # row-wise, so slicing only lowers peak memory
            pl.concat([normalize(raw.slice(a, NORM_SLICE), maps) for a in range(0, len(raw), NORM_SLICE)]).write_parquet(tmp)
            del raw
            tmp.replace(out)
            _log(f"normalised {split} source {s}")


def step_candidates(work: Path, splits: list[str], only: list[str] | None = None) -> None:
    for split in splits:
        _log(f"building candidates for {split}")
        candidates.build(norm_paths(work, split), work / split, only=only)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prepare", "candidates", "enrich", "train", "tune", "predict", "stage2", "xval", "xval_st", "xval_ablate", "phantom", "all"])
    ap.add_argument("--data-dir", type=Path, default=Path("../student_resource/dataset"))
    ap.add_argument("--work-dir", type=Path, default=Path("../work"))
    ap.add_argument("--out-dir", type=Path, default=Path("../output"))
    ap.add_argument("--splits", default="train,test", help="for the candidates step")
    ap.add_argument("--countries", default=None, help="candidates step: only (re)build these countries")
    args = ap.parse_args()
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    if args.step in ("prepare", "all"):
        step_prepare(args.data_dir, work)
    if args.step in ("candidates", "all"):
        step_candidates(work, args.splits.split(","), args.countries.split(",") if args.countries else None)
    if args.step == "enrich":
        for split in args.splits.split(","):
            enrich.enrich(work, split)
    if args.step == "train":
        model.step_train(args.data_dir, work)
    if args.step == "tune":
        model.step_tune(args.data_dir, work)
    if args.step == "predict":
        model.step_predict(args.data_dir, work, args.out_dir)
    if args.step == "xval":
        experiments.cross_country(args.data_dir, work)
    if args.step == "xval_ablate":
        experiments.cross_country(args.data_dir, work, experiments.ablation_groups, "xval_ablate_report.json")
    if args.step == "xval_st":
        experiments.cross_country_selftrain(args.data_dir, work)
    if args.step in ("stage2", "all"):
        stage2.step_stage2(args.data_dir, work, args.out_dir)
    if args.step == "phantom":
        phantom.run(args.data_dir, work, args.out_dir)


if __name__ == "__main__":
    main()

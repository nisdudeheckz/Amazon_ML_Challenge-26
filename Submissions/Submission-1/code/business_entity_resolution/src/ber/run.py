"""End-to-end CLI.

    python -m ber.run all        --data-dir DATA --work-dir WORK --out-dir OUT
    python -m ber.run prepare    ...   # learn transliteration maps, normalise all sources
    python -m ber.run candidates ...   # blocking + pair features for train and test
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

from . import candidates, model, translit
from .io import read_ground_truth, read_source
from .normalize import normalize


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def norm_paths(work: Path, split: str) -> dict[int, Path]:
    return {s: work / "norm" / f"{split}_s{s}.parquet" for s in (1, 2, 3)}


def step_prepare(data: Path, work: Path) -> None:
    (work / "norm").mkdir(parents=True, exist_ok=True)
    raw = {s: read_source(data / "train" / f"train_source{s}.tsv") for s in (1, 2, 3)}
    pairs = read_ground_truth(data / "train" / "train_ground_truth.tsv")
    maps = translit.learn(raw[1], pl.concat([raw[2], raw[3]]), pairs)
    translit.save(maps, work / "translit.json")
    _log(f"learned transliteration maps: {len(maps['name'])} name tokens, {len(maps['addr'])} address components")
    for s in (1, 2, 3):
        normalize(raw[s], maps).write_parquet(norm_paths(work, "train")[s])
    del raw
    for s in (1, 2, 3):
        normalize(read_source(data / "test" / f"test_source{s}.tsv"), maps).write_parquet(norm_paths(work, "test")[s])
        _log(f"normalised test source {s}")


def step_candidates(work: Path, splits: list[str], only: list[str] | None = None) -> None:
    for split in splits:
        _log(f"building candidates for {split}")
        candidates.build(norm_paths(work, split), work / split, only=only)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prepare", "candidates", "train", "tune", "predict", "all"])
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
    if args.step in ("train", "all"):
        model.step_train(args.data_dir, work)
    if args.step == "tune":
        model.step_tune(args.data_dir, work)
    if args.step in ("predict", "all"):
        model.step_predict(args.data_dir, work, args.out_dir)


if __name__ == "__main__":
    main()

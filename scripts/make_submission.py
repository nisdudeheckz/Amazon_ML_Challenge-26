"""Package a submission into Submissions/Submission-<n>/ (next free n unless --n given).

Layout (mirrors the required zip structure):

    Submissions/Submission-<n>/
    ├── output/
    │   ├── matching_results.tsv
    │   └── candidate_pairs.tsv
    ├── code/business_entity_resolution/
    │   ├── src/ber/*.py
    │   ├── README.md
    │   └── requirements.txt
    ├── Documentation_template.md
    ├── metrics.json               (validation report; not part of the zip)
    ├── validation.txt             (output of utils/validate_submission.py; not zipped)
    └── <team>_submission.zip

Usage (from the repo root):
    python scripts/make_submission.py --team MyTeam [--n 3] [--output-dir output] [--work-dir work]
                                      [--src src] [--doc docs/Documentation_template.md] [--report work/x.json]
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def next_n(base: Path) -> int:
    ns = [int(m.group(1)) for p in base.glob("Submission-*") if (m := re.fullmatch(r"Submission-(\d+)", p.name))]
    return max(ns, default=0) + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="TEAM_NAME")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "output")
    ap.add_argument("--work-dir", type=Path, default=ROOT / "work")
    ap.add_argument("--src", type=Path, default=ROOT / "src", help="source tree to ship (default: src/)")
    ap.add_argument("--doc", type=Path, default=ROOT / "docs" / "Documentation_template.md")
    ap.add_argument("--report", type=Path, default=None, help="metrics json to copy (default: work/train_report.json)")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    base = ROOT / "Submissions"
    base.mkdir(exist_ok=True)
    n = args.n or next_n(base)
    dest = base / f"Submission-{n}"
    if dest.exists():
        sys.exit(f"{dest} already exists; pass a different --n")
    print(f"creating {dest}")

    out = dest / "output"
    out.mkdir(parents=True)
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        shutil.copy2(args.output_dir / f, out / f)

    code = dest / "code" / "business_entity_resolution"
    shutil.copytree(args.src, code / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(ROOT / "docs" / "CODE_README.md", code / "README.md")
    shutil.copy2(ROOT / "requirements.txt", code / "requirements.txt")
    shutil.copy2(args.doc, dest / "Documentation_template.md")
    report = args.report or args.work_dir / "train_report.json"
    if report.exists():
        shutil.copy2(report, dest / "metrics.json")

    validator = ROOT / "student_resource" / "utils" / "validate_submission.py"
    res = subprocess.run(
        [sys.executable, str(validator), "--matching", str(out / "matching_results.tsv"),
         "--candidate", str(out / "candidate_pairs.tsv"),
         "--test-dir", str(ROOT / "student_resource" / "dataset" / "test"), "--check-ids"],
        capture_output=True, text=True,
    )
    (dest / "validation.txt").write_text(res.stdout + res.stderr, encoding="utf-8")
    print(res.stdout.strip()[-2000:])
    if res.returncode != 0:
        print("!! validator reported problems (see validation.txt)")

    if not args.no_zip:
        zpath = dest / f"{args.team}_submission.zip"
        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for p in sorted(dest.rglob("*")):
                rel = p.relative_to(dest)
                if p.is_dir() or p == zpath or rel.parts[0] not in ("output", "code", "Documentation_template.md"):
                    continue
                z.write(p, rel.as_posix())
        print(f"wrote {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

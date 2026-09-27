"""Pick the final submission from a finished run, by a rule fixed before seeing results.

    python scripts/pick_final.py --work-dir W --output-dir O --src S --resource-dir R [--package]

Candidates (all from the same run):
    root    current stage-2 model + its decision rule         (O/)
    rootph  same model, decision re-tuned with orphan records (O/rootph/)
    ph      stage 2 refitted with orphan records              (O/ph/)

Rule:
  1. root must reach the champion's recorded held-out F0.5 (BAR) within ROOT_TOL, or
     nothing new is submitted (Submission-5 stays).
  2. A phantom variant replaces root only if, on the orphan-record held-out, it beats root
     by >= MIN_GAIN, and on the normal held-out it loses <= MAX_NORMAL_LOSS.
Writes <work>/final_decision.json; with --package also builds Submission-<n> via
make_submission.py (validator included).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

BAR = 0.98859            # champion family held-out F0.5 (Submission-4/5 records)
ROOT_TOL = 0.0003
MIN_GAIN = 0.0005
MAX_NORMAL_LOSS = 0.0005
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--resource-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--team", default="TEAM_NAME")
    ap.add_argument("--package", action="store_true")
    a = ap.parse_args()

    s2 = json.load(open(a.work_dir / "stage2_report.json"))
    root_norm = s2["metrics"]["f05"]
    ph_path = a.work_dir / "phantom_report.json"
    ph = json.load(open(ph_path)) if ph_path.exists() else None
    rows = {"root": {"normal": root_norm,
                     "phantom": ph["root_on_phantom"]["all"]["f05"] if ph else None, "dir": a.output_dir}}
    if ph:
        rows["rootph"] = {"normal": ph["rootph_on_normal"]["all"]["f05"],
                          "phantom": ph["rootph_on_phantom"]["all"]["f05"], "dir": a.output_dir / "rootph"}
        rows["ph"] = {"normal": ph["ph_on_normal"]["all"]["f05"],
                      "phantom": ph["ph_on_phantom"]["all"]["f05"], "dir": a.output_dir / "ph"}

    choice, why = "champion", f"root held-out {root_norm:.5f} < {BAR - ROOT_TOL:.5f}: keep Submission-5"
    if root_norm >= BAR - ROOT_TOL:
        choice, why = "root", f"root held-out {root_norm:.5f} >= {BAR - ROOT_TOL:.5f}"
        if ph:
            base = rows["root"]["phantom"]
            ok = {k: r for k, r in rows.items() if k != "root" and (r["dir"] / "matching_results.tsv").exists()
                  and r["phantom"] >= base + MIN_GAIN and r["normal"] >= root_norm - MAX_NORMAL_LOSS}
            if ok:
                choice = max(ok, key=lambda k: ok[k]["phantom"])
                r = ok[choice]
                why += (f"; {choice} orphan-held-out {r['phantom']:.5f} vs root {base:.5f} "
                        f"(+{r['phantom'] - base:.5f}), normal {r['normal']:.5f}")
            else:
                why += "; no phantom variant cleared the gain/loss bars"
        else:
            why += "; phantom step unavailable"

    table = {k: {"normal_f05": r["normal"], "orphan_f05": r["phantom"], "output": str(r["dir"])} for k, r in rows.items()}
    decision = {"choice": choice, "reason": why, "bar": BAR, "candidates": table}
    json.dump(decision, open(a.work_dir / "final_decision.json", "w"), indent=1)
    print(json.dumps(decision, indent=1))

    if a.package and choice != "champion":
        report = a.work_dir / "final_metrics.json"
        json.dump({"decision": decision, "stage2_report": s2, "phantom_report": ph}, open(report, "w"), indent=1)
        cmd = [sys.executable, str(ROOT / "scripts" / "make_submission.py"), "--team", a.team, "--n", str(a.n),
               "--output-dir", str(rows[choice]["dir"]), "--work-dir", str(a.work_dir), "--src", str(a.src),
               "--report", str(report), "--resource-dir", str(a.resource_dir)]
        sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()

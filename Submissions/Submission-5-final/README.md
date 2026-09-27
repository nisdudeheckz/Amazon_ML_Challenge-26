# Final submission: Submission-5 outputs (champion)

- **Outputs:** the Submission-5 files, byte-verified against `champion/sub5/SHA256SUMS`.
- **Code:** the `Submissions/Submission-5/code` snapshot.
- **Documentation:** Submission-4's write-up, the same pipeline family.
- **Validator:** PASS.

**Why S5 is final:**
- S6, the stage 2 refitted with orphan records, scored below S5 on the leaderboard.
- On the normal held-out (`scripts/ensemble_check.py`):

| model | F0.5 |
|---|---|
| S5-like refit | 0.98855 |
| root (S7) | 0.98850 |
| union | 0.98846 |
| intersection | 0.98860 |

Every difference is within noise, so no combination is justified over the incumbent.

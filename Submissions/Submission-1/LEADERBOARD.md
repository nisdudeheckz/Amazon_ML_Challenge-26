# Submission-1 — leaderboard result

| | |
|---|---|
| Public leaderboard F0.5 | **0.962626** (rank 143, 2026-09-25) |
| Held-out validation F0.5 | 0.96886 (US 0.9747, India 0.9601; no France labels) |
| Top-10 threshold at the time | ≈ 0.981 (rank 1: 0.985884) |

The leaderboard is ≈0.006 below the held-out estimate. Weighting the held-out US/India
scores by the test-set composition (India 46.8%, US 38.3%, France 15.0% of S1) implies
France ≈ 0.94.

Main losses identified afterwards: blocking recall (4.6% of true pairs never became
candidates because common tokens were pruned from the retrieval product), then classifier
recall (2.7%). These are addressed in Submission-2.

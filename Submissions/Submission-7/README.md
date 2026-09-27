# Submission-7: current code, standard stage 2 (leaderboard probe)

This is the root model from the same local run as S6, with the normal held-out decision rule (expected-F0.5, floor 0.4). It adds S5's successors: address-structure and commonness-percentile features, and absolute name frequencies dropped.

- **Held-out F0.5:** 0.98850 (S5 family 0.98859). P 0.9980, R 0.9690.
- **Validator:** PASS.

**Against S5 on test (pairs):**

| country | S5 | S7 | dropped | added |
|---|---|---|---|---|
| France | 880,595 | 875,659 | 14,472 | 9,536 |
| India | 2,725,217 | 2,723,870 | 10,433 | 9,086 |
| US | 2,259,065 | 2,278,075 | 9,127 | 28,137 |

S6 scored below S5 on the leaderboard: it dropped 21.8k France pairs and added 5.8k. S7 reshuffles France less (net −4.9k, against S6's −16k) and adds recall in the US.

Treat S7 as a probe only. Keep S5 as the final unless S7's public score beats S5's.

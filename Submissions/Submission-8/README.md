# Submission-8: average of two stage-2 models (root + S5-like)

The stage-2 probabilities of the root model and a refit on S5's feature set are averaged. The expected-F0.5 rule is tuned on the held-out (floor 0.65), with `scripts/s8_ensemble.py`.

| held-out | F0.5 | singletons | P | R |
|---|---|---|---|---|
| root (S7) | 0.98850 | 0.9868 | 0.9980 | 0.9690 |
| S5-like | 0.98855 | 0.9875 | 0.9981 | 0.9692 |
| **S8 average** | **0.98861** | **0.9927** | 0.9982 | 0.9688 |

Validator: PASS. The gain is small (+0.00006 over S5-like) and mostly on singletons. Recall is unchanged, so S8 is not S6's over-conservative regime.

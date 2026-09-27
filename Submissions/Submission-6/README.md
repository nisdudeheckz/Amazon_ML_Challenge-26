# Submission-6 — orphan-record-trained stage 2

**Files:** `output/matching_results.tsv`, `output/candidate_pairs.tsv`, and `TEAM_NAME_submission.zip`. Rename the zip to `<team>_submission.zip` and fill in the team name in `Documentation_template.md`. The official validator passes (`validation.txt`).

## Why

Test has **5.5–5.8 S2/S3 records per S1** in every country, against **4.68 in train**. The number of true matches per S1 appears unchanged, at about 3.4.

The likely cause is a subsampled test S1 set. The records of the dropped entities stay in the data as orphans with no true match, and their best wrong candidate wins unopposed.

We simulated this on train by removing 18% of S1 entities and recomputing all candidate-list features (`python -m ber.run phantom`).

## Held-out results (macro F0.5, same entity-disjoint split as Submissions 2–5)

| model | normal held-out | with orphan records (test-like) |
|---|---|---|
| Submission-4/5 family (recorded) | 0.98859 | — |
| current code, standard stage 2 | 0.98850 | 0.98729 (P 0.9960) |
| … decision re-tuned with orphans | 0.98850 | 0.98732 |
| **stage 2 refitted with orphans (this submission)** | **0.98841** | **0.98783** (P 0.9975) |

The winner was picked by a rule fixed before the run (`scripts/pick_final.py`):
- it must beat the standard model by at least 0.0005 in the test-like setting;
- it must lose no more than 0.0005 on the normal held-out;
- the current code must match the champion within 0.0003.

## Against Submission-5 on test

- **Candidate pairs:** identical, same byte size.
- **Matched pairs:** 97.5–99.4% are shared with Submission-5.
- **France:** 21.8k of Submission-5's pairs are dropped and 5.8k are added. France is where false merges were suspected.
- **India:** 16.3k dropped, 5.9k added.
- **US:** 13.7k dropped, 21.8k added.

## Expected effect and risk

- **Expected leaderboard change:** about +0.0005, if test behaves like the simulation. This is modest, not a breakthrough. It targets a measured mechanism: false merges roughly double with orphan records.
- **Risk:** on data without orphans it costs −0.00009, which is noise. If the orphan explanation is wrong, the expected change is about zero.

Submission-5 remains the fallback (`champion/sub5/`, read-only).

## France risk (found after packaging; read before choosing S6 over S5)

- S6 is more conservative everywhere, but **France is hit hardest**: 21.8k of S5's France pairs are dropped (2.5%), against about 0.6% in US and India. **1,099 France S1s go from matched to empty.**
- On the normal held-out, the pairs S6 drops relative to the standard model are about **78% true**, going by the P/R arithmetic. S6 only breaks even there because removing an FP is worth about 3× keeping a TP.
- The dropped France pairs we sampled are mixed:
  - distractor-like: "Libre Club SARL" / "Libre Ecole SARL", and SA→SASU with a changed house number;
  - true-copy-like: "VUA Societe SARL" / "VUTA Societe SARL", "@ROUBAIXAMICALE", and generic-word swaps, which Submission-4 learned to accept and gained leaderboard points for.
- **Only the leaderboard can settle France.** If S6's public score is below S5's, submit S5 as the final.

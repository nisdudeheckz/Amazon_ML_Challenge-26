# Submission-3 — diagnostic probe (not a final candidate)

`output/matching_results.tsv` combines:

- Submission-2's rows for **US and India** S1 entities;
- Submission-1's rows for **France** S1 entities.

**Why.** Submission-2 scored 0.976497 on the leaderboard, against a held-out estimate of 0.9885. If US and India score as they do on held-out, France must be ≈ 0.91, down from ≈ 0.94 in Submission-1. France is also the only country where Submission-2 matches fewer records than Submission-1.

The pairs Submission-2 dropped show a clear pattern: identical address and distinctive name word, but one generic French word swapped or added ("Poker Sportive EURL" → "Poker EURL [France]", "Micro Rock Pharmacie" → "Micro Rock Fetes"). These look like true copies. Submission-2's new "vocabulary-word substitution" features, learned on US/India, seem to reject them.

**Reading the result:**

- Higher than 0.976497: Submission-1's France predictions were better. This confirms that some Submission-2 features transfer poorly to an unseen country.
- About the same or lower: the France drop has another cause.

This file only exists to measure that difference. A final solution must not treat France specially: the country is an open set, and a per-country switch would break the challenge rule. The fix therefore goes into the general pipeline.

It was built from the two committed submissions' outputs by selecting rows by the S1 entity's country label.

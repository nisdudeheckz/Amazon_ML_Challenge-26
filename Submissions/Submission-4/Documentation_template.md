# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** 2026-09-26 (Submission-4)

---

## 1. Executive Summary

Every Source-2/Source-3 record is resolved to **at most one** Source-1 entity. The pipeline has four stages:

1. Rule-based, country-agnostic normalisation, enriched with a native-script→Latin token dictionary **learned from the training ground truth**.
2. Per-country **TF-IDF top-K retrieval** over nine hashed feature namespaces. It includes **compound namespaces** (name-token pairs, name×address tokens, address bigrams) that keep candidate recall high while common tokens are pruned from the sparse product.
3. A **two-stage LightGBM**:
   - pair features plus *difference-type* features (typo vs. vocabulary substitution, legal-form changes, number mutations);
   - out-of-fold stage-1 probabilities feeding stage-2 *competition* features.
4. A one-to-one assignment with an expected-F0.5 decision rule.

On 110,216 held-out training S1 entities, macro **F0.5 = 0.9885** (US 0.9891, India 0.9876), with pair precision 99.80% and pair recall 96.92%. Submission-1 scored 0.9689 held-out and 0.9626 on the leaderboard.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA on the 2.2M / 5.0M / 5.3M training records (S1 / S2 / S3) gave the facts the design is built on:

| Observation | Consequence |
|---|---|
| 7,638,365 matched S2/S3 ids, **all distinct**: every S2/S3 record belongs to **at most one** S1 entity | Match from the S2/S3 side: retrieve S1 candidates per record, then assign each record to its single best S1 (argmax). This removes a whole class of false merges. |
| ~26% of S2/S3 records match **no** S1 entity (distractors); 5.6% of S1 entities are singletons | A calibrated "no match" decision is essential; singletons are worth a full 1.0. |
| Matched records **always share the country label** | Blocking is done per country label. The label set is taken from the data, so France is handled like any other value. It is never hard-coded and never used as a feature. |
| **38% of S1 names are duplicated** across different businesses, and the name vocabulary is small (≈1,300 distinct words in Indian names) | Name alone is not identifying, and single name tokens are all "common". Evidence comes from *combinations* (name + city, name-word pairs) and from the address. |
| **Distractors are mutations of real S1 entities**, and their mutations differ in kind from the noise applied to genuine copies (see §4.2) | Besides *how similar* a pair is, the model needs features describing *what kind of difference* separates it. |
| 18–24% of Indian S2/S3 names are written in Indic scripts (Devanagari, Gujarati, Telugu, Tamil, Kannada, Bengali, Odia, Gurmukhi…); state names in addresses too | Generic transliteration turns "East Logistics" into "ist lojistiks". A token dictionary learned from training pairs fixes this (§3.1). |
| Name noise: legal-suffix swaps, honorifics, alias markers (`DBA`, `d/b/a`, `formerly`, `trading as`, `***`) with the true name after a random prefix, domain-style names (`healthoncology.com`), leetspeak (`C0nsultants`), word shuffles, typos, appended phone numbers | Normalisation handles each explicitly. Similarity features are order-insensitive and typo-tolerant. |
| Address noise: casing, component re-ordering, abbreviations, state code ↔ name, zero-padded / truncated house numbers (`09315`, `4712→471`), dropped components, `null`, unit codes (`A-304`, `203-Q`, `B-5/8`) | Tokenised, canonicalised addresses. Numbers and unit codes are kept as separate evidence. |
| France (test only): S1 uses *regions* ("Nouvelle-Aquitaine"), S2/S3 often *departments* ("Gironde"); `N°` appears as "Ndeg" after transliteration | IDF is computed over the whole unlabelled corpus of the country, so source-specific tokens are not over-weighted. `ndeg` is an address stop-word. |

### 2.2 Solution Strategy

**Approach Type:** Blocking (sparse TF-IDF retrieval) + two-stage gradient-boosted pair classifier + constrained assignment.

**Core Innovations:**

- **(a)** Query-side retrieval that exploits the at-most-one-S1 structure.
- **(b)** Compound blocking namespaces. They cut missed true pairs from 4.6% to 1.4% while *reducing* candidates per record from 3.4 to 2.1.
- **(c)** Difference-type features that tell distractor mutations from genuine noise.
- **(d)** Out-of-fold stage-1 probabilities turned into record-level and entity-level competition features.
- **(e)** A native-script token dictionary learned from ground-truth alignments.
- **(f)** An expected-F0.5 decision rule.

---

## 3. Candidate Generation (Blocking)

### 3.1 Normalisation (all countries, same rules)

**Transliteration.** `translit.py` aligns the tokens of Indic-script S2/S3 names with the matched S1 name, position by position. Token counts agree in 99.998% of 551k training pairs. For each native token it keeps the majority Latin spelling, which gives 1,347 tokens and 96.4% token coverage on test. Native-script address components (states) are mapped to the S1 component they co-occur with in ≥80% of pairs. Anything else falls back to `anyascii`, which also handles French accents.

**Names.**

- lower-case;
- strip phone numbers, `www.`, TLDs and punctuation;
- `&` → `and`;
- undo leetspeak;
- split alias parts;
- canonicalise legal forms.

The *core* name then drops legal forms (including French SARL/SAS/SCI/EURL…), honorifics and stop-words.

**Addresses.**

- remove `null`;
- split letters from digits;
- canonicalise street types (US: Street/St, Road/Rd…; French: R./Rue, Bd, Av, Chem…);
- strip leading zeros from numbers;
- keep the house number;
- keep unit / plot codes (`203-Q`→`203q`, plus a digits-only variant, `C-24-144/4/1`→`2414441`).

### 3.2 Retrieval

For each country, every record becomes nine hashed, idf-weighted, per-namespace L2-normalised sparse sub-vectors:

| namespace | content | weight |
|---|---|---|
| `nw` | name words (all alias parts) | 0.15 |
| `ng` | name character 3-grams | 0.15 |
| `aw` | address words | 0.20 |
| `an` | address numbers | 0 (feature only) |
| `ac` | unit / plot codes | 0.10 |
| `ap` | (house number, street word) | 0.10 |
| `np` | unordered pairs of core name tokens | 0.10 |
| `xn` | core name token × address token (name + city key) | 0.15 |
| `ab` | adjacent address-token pairs | 0.05 |

The blocking score is Σ weight × cosine. IDF is computed over S1+S2+S3 of the country (unlabelled). Features that no S1 record has are excluded from the vectors; their share is kept as a feature. Features with S1 document frequency > 6,000 are left out of the sparse product but still count in the norms. `sparse_dot_topn` computes the top-20 on 14 threads. Each country runs in its own process, and S1 feature statistics are held as sorted NumPy arrays, which keeps memory at about 8 GB.

**Why compound namespaces.** Pruning common tokens was the single largest loss in Submission-1. A record whose identity is a *combination* of common tokens could not be retrieved: "Grogancell.Com | 44 Saint, Brooklyn" vs "Grogan Cell | 1746 44 Street, Brooklyn", or Indian names built from a 1,300-word vocabulary. Raising the pruning threshold alone fixes this but is 12× slower. Compound features are rare, so they survive pruning:

| (60k-record samples) | US recall kept | India recall kept | top-K time |
|---|---|---|---|
| Submission-1 (6 namespaces, max_df 3,000) | 96.4% | 93.6% | 6 s |
| no pruning at all | 99.0% | 98.4% | 300–455 s |
| **Submission-2** (9 namespaces, max_df 6,000) | **98.8%** | **98.1%** | 18–22 s |

**Summary**

- **Blocking keys used:** hashed TF-IDF features over the nine namespaces above; there are no exact-key blocks.
- **Candidates kept:** rank < 10 and score ≥ 0.6 × the record's best score. This is exactly the set the model scores and that `candidate_pairs.tsv` contains.
- **Candidate pairs generated:**
  - train: 20.9M (2.0 per S2/S3 record; Submission-1 had 34.5M);
  - test: 22,542,150 (2.3 per record).

  The reduction ratio vs. the within-country Cartesian product is > 99.9998%.
- **How true matches were kept:** the train candidate set contains **98.6% of all ground-truth pairs** (US 98.9%, India 98.2%). Submission-1's set held 95.4%. The remaining misses are mostly empty-address records whose name is also changed.

---

## 4. Matching Model

### 4.1 Pair features (134, all country-agnostic)

**Per-namespace sparse similarity.** For each of the nine namespaces:

- cosine, overlap count and Jaccard;
- non-zero counts on each side;
- the fraction of the record's features unseen in S1.

**Strings.** On core names and addresses, via rapidfuzz `cpdist`:

- `ratio`, `token_sort_ratio`, `token_set_ratio`, `partial_ratio`;
- Jaro-Winkler of space-less names;
- best alias-part match;
- full-name ratio.

**Numbers.**

- house-number equality and Jaro-Winkler;
- Levenshtein distance;
- log absolute numeric difference;
- prefix/suffix relation;
- length difference.

**Context.**

- blocking score and rank; the record's best and second-best scores; gap and ratio to the best;
- S1-side competition: how many records retrieve the same S1, how many have it at rank 1, this record's rank and score gap among them;
- name rarity: how many S1 records share the exact core name;
- lengths, token and component counts;
- native-script flag;
- source (S2 / S3).

### 4.2 Difference-type features (`enrich.py`)

Held-out errors showed that distractors are generated by mutating real entities, in ways unlike the noise on genuine copies:

| | distractor (different business) | true copy (same business) |
|---|---|---|
| name | a word replaced by another *real vocabulary word* (Tech→Leather, Tourism→Aluminium) | *character typos* (Enerbhg, Tnaribesg); words dropped or shuffled |
| legal form | switched (Private Ltd→**Public** Ltd, Ltd→Private Ltd) | abbreviated, dropped, re-ordered |
| numbers | small arithmetic change (12590→12594, 102/2A→102/7A) | truncation or digit typo (4712→471, 2A/84→2A/8) |

For the unmatched name tokens on each side, the features record:

- how many are typos of a counterpart (Jaro-Winkler ≥ 0.8);
- how many are *known vocabulary words* (used by ≥ 20 S1 names);
- how many are rare;
- the log max document frequency and the minimum Jaro-Winkler.

They also cover one-sided legal forms (count, per-form flags for public/pvt/ltd/llc/inc/corp/co/pc, and equality) and the closest unit-code pair (minimum Levenshtein, prefix/suffix relation, and single same-length substitution). On train pairs, "unmatched record name token is a known vocabulary word" averages 0.34 for non-matches vs 0.05 for matches. "Public only on the S2/S3 side" occurs in 1.6% of non-matches and 0% of matches.

### 4.3 Two-stage model

**Stage 1.** A LightGBM (255 leaves, learning rate 0.08, feature/bagging fraction 0.7/0.8, 1,000 rounds) is fitted twice, on the two halves of a 40% fitting sample split by a hash of the S2/S3 id. Every train and test pair is scored by the model of the *other* half. Stage-1 probabilities are therefore out-of-fold on train and identically distributed on test.

**Stage 2.** A LightGBM on the 134 pair features, the S1 context, p1 and competition features:

- **Record level:** rank of p1 among the record's candidates; best and second-best p1; sum, share and margin.
- **Entity level:** sum and max of p1 over all records claiming the S1; count with p1 > 0.5; this record's rank; the same counts restricted to records whose *best* candidate it is.

It has 150 features, uses 6.45M fitting pairs, and early-stopped at 222 rounds. The top gains are record-level margin, p1, max, share and second-best p1, then the S1-side claim counts.

**Decision.** Each S2/S3 record goes only to its highest-probability S1 (one-to-one). Two rules are tuned on held-out S1 entities by macro F0.5:

- **(i)** a global threshold;
- **(ii)** per-entity expected F0.5: keep the top-m assigned records maximising `1.25·Σp_i / (m + 0.25·(Σp + λ))`, or none if `Π(1−p_i)·e^{−λ}` is higher.

Rule (ii) with λ = 0 is used: 0.98852, vs 0.98840 for the best threshold t = 0.725.

---

## 5. Results & Error Analysis

**Validation protocol.** 5% of training S1 entities (hash-selected, 110,216) are held out. Every S2/S3 record that has any held-out entity among its candidates is excluded from fitting, so held-out predictions are complete and out-of-sample. Retrieval uses the *full* training S1 index, so candidate density matches the test setting.

| | macro F0.5 | singletons | non-singletons | pair P | pair R |
|---|---|---|---|---|---|
| Submission-1 (single stage, 6 namespaces) | 0.9689 | 0.9681 | 0.9689 | 99.44% | 92.92% |
| Submission-2 stage 1 (out-of-fold) | 0.9864 | 0.9873 | 0.9863 | 99.69% | 96.53% |
| **Submission-2 stage 2** | **0.9885** | 0.9875 | 0.9886 | **99.80%** | **96.92%** |
| — US | 0.9891 | 0.9866 | 0.9893 | 99.82% | 97.08% |
| — India | 0.9876 | 0.9889 | 0.9876 | 99.78% | 96.68% |

- **F_0.5 Score (macro, held-out):** 0.9885
- **Common false positives (wrong merges):**
  - distractors whose mutation looks like noise: a double character typo in a name word ("Aditraj Makers" → "Aditrade Mares"), or a house number changed by one digit;
  - empty-address records whose name is shared by two S1 entities.
- **Common false negatives (missed matches):**
  - ≈1.4% of true pairs are never candidates: empty address plus altered name, or a junk replacement name plus a truncated address;
  - true copies whose house number was replaced rather than truncated ("5 Hill St" → "168 Hill St").

**Leaderboard.** Submission-1 scored 0.9626 (held-out 0.9689). The gap is attributed mainly to France, which has no training labels: the implied France score was ≈ 0.94. Submission-2 adds corpus-level IDF and French address stop-words for it.

---

## 6. Conclusion

Most of the gain over Submission-1 came from understanding *why* errors happen rather than from a bigger model:

- Blocking lost pairs whose evidence is a combination of common tokens; compound namespaces fixed that cheaply.
- Distractors are mutations of real businesses; difference-type features and stage-2 competition features let the classifier recognise them.

Together these raised held-out macro F0.5 from 0.9689 to 0.9885 at 99.8% pair precision.


---

## Addendum — Submission-4 (changes since Submission-2)

**Leaderboard feedback.** Submission-2 scored 0.976497, while held-out validation gave 0.9885.
- If US and India score as they do on held-out, the implied France score is ≈ 0.91, down from ≈ 0.94 in Submission-1.
- France was also the only country where Submission-2 matched fewer records than Submission-1.

**Cross-country validation.** France has no labels, so I trained on one training country, tuned the decision rule on that country's held-out entities, and scored the *other* country. This is exactly how an unseen country is handled.

| features | India → US (same / cross) | US → India (same / cross) |
|---|---|---|
| base pair features | 0.9807 / 0.9671 | 0.9851 / 0.9282 |
| + legal-form & unit-code differences | 0.9847 / 0.9674 | 0.9866 / 0.9369 |
| + per-country noise-ratio name features | 0.9858 / 0.9697 | 0.9871 / **0.9484** |
| + absolute-frequency name features (as in Submission-2) | 0.9860 / 0.9698 | 0.9872 / 0.9474 |

Transfer to an unseen country costs mostly **precision**.

**New per-country noise-ratio features.** Noise-inserted name words are strongly over-represented in S2/S3 relative to S1 in every country:
- US: "Greater", "Midtown", "District";
- France: "Holding", "Participations", "Développement".

Ordinary vocabulary sits at a ratio of 0.82–0.89 everywhere. Unmatched name tokens are now classified as:
- typos;
- inserted noise words (ratio ≥ 3);
- vocabulary substitutions.

The ratio is computed per country, without labels, so it transfers. It gave the largest cross-country gain in the table above: +2.0 points US → India.

**Run.** The pipeline ran on AWS EC2 (r7i.2xlarge) from Submission-2's candidates, with the new difference features and stage-2 model:
- held-out macro F0.5 **0.98859** (stage-1 out-of-fold 0.98647);
- test output: 5,845,108 matches for 1,633,123 S1 entities;
- records matched: France 60.9%, India 57.6%, US 59.1%.

---

## Appendix

### A. Code Artefacts

The code lives in `code/business_entity_resolution/src/ber/`. The entry point, run from `src/`, is:

```
python -m ber.run all --data-dir DATA --work-dir ../work --out-dir ../output
```

This runs `prepare` → `candidates` → `stage2`, and `stage2` runs `enrich` itself when needed.

| module | role |
|---|---|
| `io` | TSV I/O with an explicit tab separator and no quoting |
| `translit` | learned transliteration |
| `normalize` | name / address normalisation |
| `vectors` | hashed namespaces, NumPy feature statistics, numba pair-cosine kernels |
| `blocking` | top-K retrieval |
| `pairfeat` | pair features |
| `enrich` | difference-type features |
| `candidates` | per-country subprocess orchestration |
| `model` | validation protocol, decision rules, single-stage variant |
| `stage2` | out-of-fold stage 1, competition features, stage-2 model, test output |
| `metrics` | exact macro F0.5 |
| `run` | CLI |

Dependencies are pinned in `requirements.txt` (all MIT/BSD/Apache-2.0/ISC). No external data, APIs, geocoders or pretrained models are used. The only models are LightGBM ensembles trained from scratch, a few MB each.

### B. Additional Results

**Most important stage-2 features (LightGBM gain):**

1. record-level p1 margin
2. p1
3. record's max p1
4. p1 share
5. number of records having this S1 at rank 1
6. p1 of the other records claiming the S1
7. second-best p1
8. sum of p1
9. S1 best-claim sum
10. blocking top-2 margin

Among the pair features, the most useful are full-name ratio, S1 name rarity, address ratio and the unseen share of name×address compound features.

**Test-set output statistics:**

- 9,969,589 S2/S3 records produced 22,542,150 candidate pairs.
- 5,759,753 records were matched, covering 1,631,647 of the 1,732,544 S1 entities.

| country | S2/S3 records matched | S1 entities with ≥1 match | avg. matches per S1 | records with uncertain best candidate (0.2 < p < 0.8) |
|---|---|---|---|---|
| France (unseen in training) | 59.9% | 94.4% | 3.32 | 5.1% |
| India | 56.7% | 94.1% | 3.31 | 2.1% |
| US | 58.2% | 94.2% | 3.35 | 3.0% |

France behaves like the training countries in volume. Its predictions are somewhat less certain, as expected for a country without training labels.

**Runtime (16 GB RAM, 16 threads):**

| step | time |
|---|---|
| normalisation | ≈ 5 min |
| candidate generation (train + test) | ≈ 3 h 25 min (most of it in the `max_df`=6,000 sparse product) |
| enrich | ≈ 2 min |
| stage-1 folds + out-of-fold scoring | ≈ 25 min |
| stage 2 + tuning + test output | ≈ 8 min |

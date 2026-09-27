# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** 2026-09-25 (Submission-1)

---

## 1. Executive Summary

Every Source-2/Source-3 record is resolved to **at most one** Source-1 entity. The pipeline has three stages:

1. Rule-based, country-agnostic normalisation, enriched with a native-script→Latin token dictionary **learned from the training ground truth**.
2. Per-country **TF-IDF top-K retrieval** of Source-1 candidates for every S2/S3 record, using hashed sparse vectors and a multi-threaded sparse top-K product.
3. A **LightGBM pair classifier** on 72 similarity and context features, followed by a one-to-one assignment and a decision rule that maximises expected F0.5.

On 110,216 held-out training S1 entities, macro **F0.5 = 0.9689**, with pair precision 99.44% and pair recall 92.92%.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA on the 2.2M / 5.0M / 5.3M training records (S1 / S2 / S3) produced the findings the design rests on:

| Observation | Consequence |
|---|---|
| 7,638,365 matched S2/S3 ids, **all distinct**: every S2/S3 record belongs to **at most one** S1 entity | Match from the S2/S3 side: retrieve S1 candidates per record and assign each record only to its single best S1 (argmax). This removes a whole class of false merges. |
| ~26% of S2/S3 records match **no** S1 entity (distractors); 5.6% of S1 entities are singletons | A calibrated "no match" decision is essential, and singletons are worth a full 1.0. |
| Matched records **always share the country label** | Blocking runs per country label. The label set comes from the data, so France is handled like any other value: never hard-coded, never a feature. |
| **38% of S1 names are duplicated** across different businesses (generic names such as "Apex Pennymac") | Name alone does not identify a business. Address evidence (house number, street, unit codes, city) and name *rarity* are needed. |
| Distractors are often **near-duplicates of real S1 entities**: same or similar name, same street, a nearby but different house number | House-number agreement is decisive; blocking must keep these confusable neighbours so the classifier can reject them. |
| 18–24% of Indian S2/S3 names are in Indic scripts (Devanagari, Gujarati, Telugu, Tamil, Kannada, Bengali, Odia, Gurmukhi…), and state names in addresses are too | Generic transliteration turns "East Logistics" into "ist lojistiks". A token dictionary learned from training pairs fixes this (see 3.1). |
| Name noise: legal-suffix swaps (Pvt/Private, Ltd/Limited, P.C., LLC…), honorifics (Mr, Dr, Shri), alias markers (`DBA`, `d/b/a`, `formerly`, `trading as`, `***`) with the true name after a random prefix, domain-style names (`healthoncology.com`, `#reebstrategic`), leetspeak (`C0nsultants`, `hea1th`), word shuffles, character typos, appended phone numbers | Normalisation handles each of these explicitly; similarity features are order-insensitive and typo-tolerant. |
| Address noise: upper-casing, component re-ordering, street-type abbreviations, state code ↔ full name, zero-padded or altered house numbers (`09315`, `1167-1169`), dropped components, `null` tokens, unit codes (`A-304`, `203-Q`, `B-5/8`) | Addresses are tokenised and canonicalised; numbers and unit codes are kept as separate evidence. |
| 3.3% of S2/S3 records have an empty address | These are decided on name evidence plus S1 name frequency. |

### 2.2 Solution Strategy

**Approach Type:** Blocking (sparse TF-IDF retrieval) + gradient-boosted pair classifier + constrained assignment

**Core Innovations:**

- **(a)** Query-side retrieval that exploits the at-most-one-S1 structure.
- **(b)** A native-script token dictionary learned from ground-truth alignments (96.4% token coverage on test).
- **(c)** Exact per-namespace sparse cosines for tens of millions of pairs, computed with a numba merge kernel.
- **(d)** A decision rule that maximises expected per-entity F0.5 instead of applying a single global threshold.

---

## 3. Candidate Generation (Blocking)

### 3.1 Normalisation (all countries, same rules)
* **Transliteration**: `translit.py` aligns the tokens of Indic-script S2/S3 names with the matched S1 name position by position; token counts agree in 99.998% of 551k training pairs. It keeps the majority Latin spelling per native token, giving 1,347 tokens. Native-script address components (states) map to the S1 component they co-occur with in ≥80% of pairs. Anything unmapped falls back to `anyascii`, which also handles French accents.
* **Names**:
  - lower-case;
  - strip phone numbers, `www.`, TLDs and punctuation;
  - `&` → `and`;
  - undo leetspeak inside alphanumeric tokens;
  - split at alias markers into *parts*;
  - canonicalise legal forms.

  Dropping legal forms (including French SARL/SAS/SCI/EURL…), honorifics and stop-words then gives the *core* name.
* **Addresses**:
  - remove `null`;
  - split letters from digits;
  - canonicalise street types (US: Street/St, Road/Rd, Drive/Dr…; French: R./Rue, Bd/Boulevard, Av/Avenue, Chem…);
  - keep numeric tokens without leading zeros, plus the first (house) number;
  - keep unit/plot *codes* (`203-Q`→`203q`, `A-304`→`a304`, `1-86`→`186`).

### 3.2 Retrieval
Blocking runs separately for each country. Every record becomes six hashed TF-IDF sub-vectors, each weighted by S1 idf and L2-normalised:

1. name words
2. name character 3-grams
3. address words
4. address numbers
5. address codes
6. (house number, street word) pairs

The blocking score is a weighted sum of the namespace cosines, with weights 0.25 / 0.20 / 0.30 / 0 / 0.10 / 0.15 in the order above, tuned for recall@K on train. Features with S1 document frequency > 3,000 are left out of the product but still count in the norms. `sparse_dot_topn` computes the product and the top-20 selection on 14 threads, at about 2 minutes per million S2/S3 records.

* **Blocking keys used:** hashed TF-IDF features over name tokens, name 3-grams, address tokens, unit codes and house-number×street pairs. There is no exact-key blocking.
* **Candidates kept:** rank < 10 and score ≥ 0.6 × the record's best score. The kept set is exactly what the model scores and exactly what `candidate_pairs.tsv` contains.
* **Candidate pairs generated:** 34.5M on train (3.4 per S2/S3 record) and 33,993,691 on test (3.4 per record). The reduction ratio versus the within-country Cartesian product is 99.9997%.
* **How true matches were kept:** the candidate set holds **95.4% of all training ground-truth pairs** (US 96.6%, India 93.6%), and 95.6% of those are the record's rank-1 candidate. Several alternatives added almost nothing: extra address-only and name-only retrievals raised recall by only 0.1–0.4 points while doubling or tripling the candidates. The remaining misses are mostly empty-address records with generic names ("Rapid Clean Sérvices") and addresses truncated to a city, which a precision-weighted metric would not reward matching anyway.

---

## 4. Matching Model

**Features used (72, all country-agnostic):**

- *Name*:
  - cosine, overlap and Jaccard for words and 3-grams;
  - rapidfuzz `ratio`, `token_sort_ratio`, `token_set_ratio` and `partial_ratio` on core names;
  - Jaro-Winkler on space-less names;
  - best alias-part match and exact-core equality;
  - legal-form agreement;
  - token counts and lengths;
  - native-script flag;
  - *name rarity*: the number of S1 records sharing the exact core name, for both sides.
- *Address*:
  - cosine, overlap and Jaccard for words, numbers, codes and house-number×street pairs;
  - `ratio`, `token_set_ratio`, `token_sort_ratio` and `partial_ratio`;
  - house-number equality and Jaro-Winkler;
  - component counts and lengths.
- *Retrieval context*: blocking score and rank; the record's best and second-best scores; the gap and ratio to the best; the number of retrieved candidates.
- *S1-side context*:
  - how many records compete for the same S1 entity;
  - how many of them have it as their top candidate;
  - this record's rank and score gap among them.
- *Source*: S2 vs S3 (they have different noise profiles).

**Model type:** LightGBM binary classifier (255 leaves, learning rate 0.08, feature/bagging fraction 0.7/0.8).

1. It is first fitted on 7.27M labelled pairs, with early stopping on held-out pairs (best iteration 957).
2. It is then refitted with the same number of rounds on 17.6M pairs: the fitting sample plus the held-out records.

**Decision / threshold selection:** each S2/S3 record is assigned only to its highest-probability S1 candidate (the one-to-one constraint). Two rules are tuned on held-out S1 entities by maximising macro F0.5 directly:

* **(i) Global threshold:** a single probability cut-off.
* **(ii) Per-entity expected F0.5:** among an entity's assigned records with p ≥ *floor*, keep the top-m that maximise `1.25·Σp_i / (m + 0.25·(Σp + λ))`, or keep none if `Π(1−p_i)·e^{−λ}` is higher.

The better rule is used on test: the per-entity expected-F0.5 rule with λ = 1.0 and floor = 0.5 (held-out 0.96886, versus 0.96854 for the best global threshold, t = 0.65).

---

## 5. Results & Error Analysis

**Validation protocol.** 5% of training S1 entities (hash-selected, 110,216 entities) are held out. Every S2/S3 record with any held-out entity among its candidates (1.49M records, 10.3M pairs) is excluded from fitting, so the held-out predictions are complete and out-of-sample. Retrieval runs against the *full* training S1 index, so candidate density matches the test setting.

- **F_0.5 Score (macro, held-out):** 0.9689
  - US 0.9747, India 0.9601
  - singletons 0.9681, non-singletons 0.9689
  - pair precision 99.44%, pair recall 92.92%
- **Precision-recall trade-off (global threshold rule):** see Appendix B. The optimum is flat between thresholds 0.60 and 0.70.
- **Common false positives (wrong merges, 1.7k pairs out of 381k true pairs on held-out):**
  - Near-duplicate distractors of a real entity on the same street with a nearby house number ("Physical Therapy Clinic LLC, 1407 Plantation Rd" vs a distractor at "1420 Plantation Rd"; "Hunger Project LLC, 96 Joyful Pl" vs "101 Joyful Pl").
  - Empty-address records whose name exists as two different S1 entities.
- **Common false negatives (missed matches):**
  - About 63% are blocking misses: empty addresses, junk replacement names ("Synyuma", "Verayumairi") combined with altered addresses, and addresses truncated to a unit code plus city.
  - The rest are true matches whose house number was perturbed ("351 S Bayly Ave" vs "353 S Bayly Ave"), so the model rejects them as possible distractors (p≈0.2–0.6).

---

## 6. Conclusion

Three things carry most of the performance: the at-most-one-S1 structure (query-side retrieval plus argmax assignment), the learned transliteration, and a rich set of cheap sparse and string features. Together they reach held-out macro F0.5 ≈ 0.9689 at >99.5% pair precision. The remaining headroom is mostly in blocking recall on records with little usable evidence, and in telling typo-level house-number noise apart from genuinely different neighbouring businesses.

---

## Appendix

### A. Code Artefacts
The package lives in `code/business_entity_resolution/src/ber/`. Run it from `src/`:

```bash
python -m ber.run prepare    --data-dir DATA --work-dir ../work
python -m ber.run candidates --data-dir DATA --work-dir ../work
python -m ber.run train      --data-dir DATA --work-dir ../work
python -m ber.run tune       --data-dir DATA --work-dir ../work
python -m ber.run predict    --data-dir DATA --work-dir ../work --out-dir ../output
```

| Module | Role |
|---|---|
| `io` | TSV I/O with an explicit tab separator and no quoting |
| `translit` | learned transliteration |
| `normalize` | name and address normalisation |
| `vectors` | hashed TF-IDF namespaces and numba pair kernels |
| `blocking` | top-K retrieval |
| `pairfeat` | pair features |
| `candidates` | per-country / per-chunk orchestration |
| `model` | LightGBM, validation protocol, decision rules, prediction |
| `metrics` | exact macro F0.5 |
| `run` | CLI |

Dependencies are pinned in `requirements.txt`; all are MIT/BSD/Apache-2.0/ISC. No external data, APIs, geocoders or pretrained models are used. The only model is the LightGBM ensemble trained from scratch, a few MB in size.

### B. Additional Results
**Global-threshold sweep on held-out S1 entities** (the one-to-one argmax assignment is always applied first):

| threshold | macro F0.5 | singletons | non-singletons | pair P | pair R |
|---|---|---|---|---|---|
| 0.30 | 0.9636 | 0.9439 | 0.9648 | 98.27% | 94.09% |
| 0.50 | 0.9675 | 0.9681 | 0.9675 | 99.00% | 93.54% |
| 0.65 | 0.9685 | 0.9787 | 0.9679 | 99.36% | 93.05% |
| 0.80 | 0.9678 | 0.9881 | 0.9666 | 99.63% | 92.34% |
| 0.95 | 0.9608 | 0.9948 | 0.9588 | 99.86% | 90.25% |
| expected-F (λ=1, floor 0.5) | **0.9689** | 0.9681 | 0.9689 | 99.44% | 92.92% |

**Most important features (LightGBM gain):**

1. blocking rank
2. score ratio to the record's best candidate
3. gap to the best candidate
4. number-token Jaccard and cosine
5. house-number Jaro-Winkler
6. full-name ratio
7. S1-side score gap
8. name `partial_ratio`
9. address `token_set_ratio`
10. top-2 margin

The retrieval-context and competition features dominate. This is consistent with the observation that distractors are near-duplicates of real entities.

**Test-set output statistics:**

- 9,969,589 S2/S3 records produced 33,993,691 candidate pairs.
- 5,706,322 records were matched, covering 1,631,292 of the 1,732,544 S1 entities.

| country | S2/S3 records matched | S1 entities with ≥1 match | avg. matches per S1 |
|---|---|---|---|
| France (unseen in training) | 62.0% | 95.1% | 3.43 |
| India | 55.0% | 93.6% | 3.20 |
| US | 58.2% | 94.4% | 3.35 |

France behaves like the training countries, so the country-agnostic features transfer.

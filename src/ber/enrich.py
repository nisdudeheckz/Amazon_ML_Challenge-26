"""Difference-type features for candidate pairs (run after `candidates`).

Distractors in this data are *mutations of real S1 entities*, and their mutations differ
in kind from the noise applied to genuine copies:

  name     distractor: a word replaced by another real vocabulary word (Tech -> Leather)
           true copy : character typos, dropped / shuffled / abbreviated words
  legal    distractor: switched to a different legal form (Private Ltd -> Public Ltd)
           true copy : legal form abbreviated, dropped or re-ordered
  numbers  distractor: small arithmetic change (12590 -> 12594, 102/2A -> 102/7A)
           true copy : truncation / digit typo (4712 -> 471, 2A/84 -> 2A/8)

So besides *how similar* a pair is, these features describe *what kind of difference*
separates it: unmatched name tokens are split into typos, inserted noise words and
vocabulary substitutions using a per-country, unlabelled S2/S3-vs-S1 frequency ratio;
one-sided legal forms are listed; the closest unit-code pair is characterised.
Output: `work/<split>/extra/<part>`.

Absolute S1 document-frequency versions of the name features (used in Submission-2)
were dropped: cross-country validation (train on one country, score the other) showed
they add nothing within a country and slightly hurt transfer, and they misfired on
generic-word swaps in the unseen test country.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import process
from rapidfuzz.distance import JaroWinkler, Levenshtein

ID_COLS = ["q_id", "s1_id"]
LEGAL = ["pvt", "ltd", "llc", "llp", "inc", "corp", "co", "pc", "lp", "plc", "pllc", "public",
         "sarl", "sas", "sa", "eurl", "sci", "snc"]
TYPO_JW = 0.8        # unmatched tokens at least this similar are treated as a typo pair
KNOWN_DF = 20        # a token used by >= KNOWN_DF S1 names counts as a vocabulary word
NOISE_LRATIO = float(np.log(3.0))   # >= 3x over-represented in S2/S3 names: an inserted noise word


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _records(work: Path, split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    cols = ["entity_id", "country", "nm_tok", "nm", "ad_code", "nm_core", "ad_first_num", "ad_tok"]
    norm = work / "norm"
    q = pl.concat([pl.read_parquet(norm / f"{split}_s{s}.parquet", columns=cols) for s in (2, 3)])
    s1 = pl.read_parquet(norm / f"{split}_s1.parquet", columns=cols)
    return q, s1


def structure_stats(q: pl.DataFrame, s1: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Unlabelled per-country structure of the split, per record.

    address density  how many S1 businesses share an address (house number + first two
                     street words), and how many also share the first name word. France
                     has ~11 S1 per shared address vs ~1.5-4 in the training countries, so
                     an address match is weaker evidence there; these let the model learn
                     from India's denser addresses how much to trust it.
    variant support  how many S2/S3 records carry exactly this name at this address. Noise
                     swaps words independently per copy, so a name variant repeated across
                     records usually belongs to a separate business (train: 2+ supporting
                     records -> 7-17% true vs 80-83% with none, for word-swap pairs).
    """
    akey = pl.col("ad_first_num").fill_null("") + "|" + pl.col("ad_tok").list.slice(0, 2).list.join(" ").fill_null("")
    nkey = (pl.col("nm_core").fill_null("") + "|" + pl.col("ad_first_num").fill_null("") + "|"
            + pl.col("ad_tok").list.first().fill_null(""))
    has_addr = akey.str.len_chars() > 3
    s = s1.select("entity_id", "country", akey.alias("a"), nkey.alias("k"),
                  pl.col("nm_tok").list.first().fill_null("").alias("w1"), has_addr.alias("ok"))
    r = q.select("entity_id", "country", akey.alias("a"), nkey.alias("k"), has_addr.alias("ok"))
    s_addr = s.filter("ok").group_by("country", "a").agg(pl.len().alias("n"))
    s_addr_w1 = s.filter("ok").group_by("country", "a", "w1").agg(pl.len().alias("n_w1"))
    q_key = r.filter("ok").group_by("country", "k").agg(pl.len().alias("nk"))
    s_out = (
        s.join(s_addr, on=["country", "a"], how="left").join(s_addr_w1, on=["country", "a", "w1"], how="left")
        .join(q_key, on=["country", "k"], how="left")
        .select(pl.col("entity_id").alias("s1_id"),
                pl.when("ok").then(pl.col("n")).cast(pl.Float32).alias("s_addr_n"),
                pl.when("ok").then(pl.col("n_w1")).cast(pl.Float32).alias("s_addr_w1_n"),
                pl.when("ok").then(pl.col("nk").fill_null(0)).cast(pl.Float32).alias("s_support"))
    )
    q_out = (
        r.join(s_addr, on=["country", "a"], how="left").join(q_key, on=["country", "k"], how="left")
        .select(pl.col("entity_id").alias("q_id"),
                pl.when("ok").then(pl.col("n").fill_null(0)).cast(pl.Float32).alias("q_addr_s1n"),
                pl.when("ok").then(pl.col("nk") - 1).cast(pl.Float32).alias("q_support"))
    )
    return q_out, s_out


def _legal(nm: pl.Expr) -> pl.Expr:
    return nm.str.split(" ").list.eval(pl.element().filter(pl.element().is_in(LEGAL))).list.unique()


def _best_cross(a: pl.DataFrame, b: pl.DataFrame, scorer, agg: str) -> pl.DataFrame:
    """For every (pid, token) of `a`, the best `scorer` value against the tokens of `b`
    with the same pid. a, b: (pid, t). Returns (pid, t, best)."""
    x = a.join(b, on="pid", suffix="_b")
    if len(x) == 0:
        return a.with_columns(pl.lit(None, pl.Float32).alias("best"))
    x = x.with_columns(pl.Series("v", process.cpdist(x["t"].to_list(), x["t_b"].to_list(),
                                                    scorer=scorer, workers=-1, dtype=np.float32)))
    fn = pl.col("v").max() if agg == "max" else pl.col("v").min()
    return x.group_by("pid", "t").agg(fn.alias("best"))


def token_stats(q: pl.DataFrame, s1: pl.DataFrame) -> pl.DataFrame:
    """Per (country, token): S1 document frequency `df` and the log ratio of the token's
    frequency among S2/S3 names to its frequency among S1 names. Ordinary vocabulary sits
    near ratio 0.8-0.9 in every country; words the noise process inserts ("[Center]",
    "Partners", "Holding", "Developpement") are several times over-represented in S2/S3.
    Unlabelled and computed per country, so it transfers to countries unseen in training."""
    def counts(df: pl.DataFrame, name: str) -> pl.DataFrame:
        return (df.select("country", pl.col("nm_tok").list.unique().alias("t")).explode("t").drop_nulls()
                .group_by("country", "t").agg(pl.len().alias(name)))
    n1 = s1.group_by("country").agg(pl.len().alias("N1"))
    nq = q.group_by("country").agg(pl.len().alias("NQ"))
    st = counts(s1, "df").join(counts(q, "nq"), on=["country", "t"], how="full", coalesce=True).fill_null(0)
    st = st.join(n1, on="country").join(nq, on="country").select(
        "country", "t", "df",
        (((pl.col("nq") + 1) / pl.col("NQ")) / ((pl.col("df") + 1) / pl.col("N1"))).log().alias("lratio"),
    )
    # Scale-free commonness: occurrence-weighted CDF of S1 document frequency within the
    # country (share of S1 name-token occurrences whose token is at most this common).
    # Train word-swap pairs with an identical address are 89-99% true when the rarer
    # swapped word is in the top fifth of this scale vs 61-77% in the bottom fifth;
    # absolute counts did not transfer (France's S1 is ~5x smaller than the US's).
    return st.sort("country", "df").with_columns(
        (pl.col("df").cum_sum().over("country") / pl.col("df").sum().over("country")).cast(pl.Float32).alias("dfpct")
    )


def pair_diff_features(pairs: pl.DataFrame, q: pl.DataFrame, s1: pl.DataFrame, vocab: pl.DataFrame) -> pl.DataFrame:
    """pairs: (q_id, s1_id). q / s1: (entity_id, country, nm_tok, nm, ad_code).
    vocab: token_stats() output (country, t, df, lratio)."""
    d = (
        pairs.select(ID_COLS).with_row_index("pid")
        .join(q.rename({"entity_id": "q_id"}), on="q_id", how="left")
        .join(s1.drop("country").rename({"entity_id": "s1_id"}), on="s1_id", how="left", suffix="_s")
        .sort("pid")
    )
    n = len(d)
    # ---- name tokens: exact matches, typo pairs, vocabulary substitutions
    d = d.with_columns(
        pl.col("nm_tok").list.set_difference(pl.col("nm_tok_s")).alias("qx"),
        pl.col("nm_tok_s").list.set_difference(pl.col("nm_tok")).alias("sx"),
    )
    qx = d.select("pid", pl.col("qx").alias("t")).explode("t", empty_as_null=True).drop_nulls("t")
    sx = d.select("pid", pl.col("sx").alias("t")).explode("t", empty_as_null=True).drop_nulls("t")
    pid_country = d.select("pid", "country")
    qb = _best_cross(qx, sx, JaroWinkler.normalized_similarity, "max").with_columns(pl.col("best").fill_null(0))
    sb = _best_cross(sx, qx, JaroWinkler.normalized_similarity, "max").with_columns(pl.col("best").fill_null(0))
    feats = {}
    for name, tb in (("q", qb), ("s", sb)):
        tb = (tb.join(pid_country, on="pid", how="left").join(vocab, on=["country", "t"], how="left")
              .with_columns(pl.col("df").fill_null(0), pl.col("lratio").fill_null(0.0), pl.col("dfpct").fill_null(0.0)))
        nontypo = pl.col("best") < TYPO_JW
        g = tb.group_by("pid").agg(
            pl.len().alias("n"),
            (pl.col("best") >= TYPO_JW).sum().alias("typo"),
            pl.col("best").min().alias("minjw"),
            (nontypo & (pl.col("lratio") >= NOISE_LRATIO)).sum().alias("noise"),
            (nontypo & (pl.col("lratio") < NOISE_LRATIO) & (pl.col("df") >= KNOWN_DF)).sum().alias("vocab"),
            pl.when(nontypo).then(pl.col("lratio")).otherwise(None).max().alias("maxlratio"),
            pl.when(nontypo).then(pl.col("dfpct")).otherwise(None).max().alias("maxpct"),
            pl.when(nontypo).then(pl.col("dfpct")).otherwise(None).min().alias("minpct"),
        )
        full = pl.DataFrame({"pid": np.arange(n, dtype=np.uint32)}).join(g, on="pid", how="left").sort("pid")
        feats[f"nm_{name}x_n"] = full["n"].fill_null(0).to_numpy().astype(np.float32)
        feats[f"nm_{name}x_typo"] = full["typo"].fill_null(0).to_numpy().astype(np.float32)
        feats[f"nm_{name}x_minjw"] = full["minjw"].to_numpy().astype(np.float32)
        feats[f"nm_{name}x_noise"] = full["noise"].fill_null(0).to_numpy().astype(np.float32)
        feats[f"nm_{name}x_vocab"] = full["vocab"].fill_null(0).to_numpy().astype(np.float32)
        feats[f"nm_{name}x_maxlratio"] = full["maxlratio"].to_numpy().astype(np.float32)
        feats[f"nm_{name}x_maxpct"] = full["maxpct"].to_numpy().astype(np.float32)
        feats[f"nm_{name}x_minpct"] = full["minpct"].to_numpy().astype(np.float32)
    # ---- legal forms present on one side only
    lg = d.select(_legal(pl.col("nm")).alias("lq"), _legal(pl.col("nm_s")).alias("ls"))
    lg = lg.with_columns(
        pl.col("lq").list.set_difference(pl.col("ls")).alias("oq"),
        pl.col("ls").list.set_difference(pl.col("lq")).alias("os"),
    )
    feats["legal_q_only_n"] = lg["oq"].list.len().to_numpy().astype(np.float32)
    feats["legal_s_only_n"] = lg["os"].list.len().to_numpy().astype(np.float32)
    feats["legal_equal"] = ((lg["oq"].list.len() == 0) & (lg["os"].list.len() == 0)
                            & (lg["lq"].list.len() > 0)).to_numpy().astype(np.float32)
    for form in ("public", "pvt", "ltd", "llc", "inc", "corp", "co", "pc"):
        feats[f"legal_q_only_{form}"] = lg["oq"].list.contains(form).to_numpy().astype(np.float32)
        feats[f"legal_s_only_{form}"] = lg["os"].list.contains(form).to_numpy().astype(np.float32)
    # ---- unit / plot codes: closest pair
    qc = d.select("pid", pl.col("ad_code").alias("t")).explode("t", empty_as_null=True).drop_nulls("t")
    sc = d.select("pid", pl.col("ad_code_s").alias("t")).explode("t", empty_as_null=True).drop_nulls("t")
    x = qc.join(sc, on="pid", suffix="_b")
    if len(x):
        x = x.with_columns(
            pl.Series("lev", process.cpdist(x["t"].to_list(), x["t_b"].to_list(), scorer=Levenshtein.distance,
                                            workers=-1, dtype=np.int32)),
            (pl.col("t").str.starts_with(pl.col("t_b")) | pl.col("t_b").str.starts_with(pl.col("t"))
             | pl.col("t").str.ends_with(pl.col("t_b")) | pl.col("t_b").str.ends_with(pl.col("t"))).alias("affix"),
        )
        g = x.group_by("pid").agg(
            pl.col("lev").min().alias("lev"),
            pl.col("affix").filter(pl.col("lev") > 0).any().alias("affix"),
            ((pl.col("lev") == 1) & (pl.col("t").str.len_chars() == pl.col("t_b").str.len_chars())).any().alias("subst1"),
        )
    else:
        g = pl.DataFrame(schema={"pid": pl.UInt32, "lev": pl.Int32, "affix": pl.Boolean, "subst1": pl.Boolean})
    full = pl.DataFrame({"pid": np.arange(n, dtype=np.uint32)}).join(g, on="pid", how="left").sort("pid")
    feats["code_minlev"] = full["lev"].cast(pl.Float32).to_numpy()
    feats["code_affix"] = full["affix"].cast(pl.Float32).to_numpy()
    feats["code_subst1"] = full["subst1"].cast(pl.Float32).to_numpy()
    return pl.concat([d.select(ID_COLS), pl.DataFrame(feats)], how="horizontal")


def enrich(work: Path, split: str) -> None:
    q, s1 = _records(work, split)
    vocab = token_stats(q, s1)
    q_struct, s_struct = structure_stats(q, s1)
    out_dir = work / split / "extra"
    out_dir.mkdir(exist_ok=True)
    for p in sorted((work / split / "parts").glob("*.parquet")):
        pairs = pl.read_parquet(p, columns=ID_COLS)
        (pair_diff_features(pairs, q, s1, vocab)
         .join(q_struct, on="q_id", how="left").join(s_struct, on="s1_id", how="left")
         .write_parquet(out_dir / p.name))
        _log(f"enriched {split}/{p.name}: {len(pairs):,} pairs")

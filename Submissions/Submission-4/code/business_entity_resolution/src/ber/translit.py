"""Learn native-script -> Latin token dictionaries from the *training* ground truth.

Indian S2/S3 names are often written in Devanagari, Gujarati, Telugu, Tamil, Kannada,
Bengali, ... scripts ("हाईटेक लॉजिस्टिक्स प्राइवेट लिमिटेड" for "Hitech Logistics Private
Limited"). In training pairs such names have the same number of tokens as the S1
name (>99.99%), so tokens can be aligned positionally and counted. The resulting
dictionary maps each native token to its majority Latin spelling. Unknown tokens
fall back to generic `anyascii` transliteration.

For addresses, fully native-script comma components (usually the state) are mapped
to the S1 address component that co-occurs with them most consistently.

Only training data is used.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

NATIVE_RE = r"[ऀ-෿]"  # Indic blocks (Devanagari ... Sinhala)
_ZW = {"‌": "", "‍": ""}


def _clean_native(tok: str) -> str:
    for k, v in _ZW.items():
        tok = tok.replace(k, v)
    return tok


def _latin(tok: str) -> str:
    return "".join(ch for ch in tok.lower() if ch.isalnum())


def learn(s1: pl.DataFrame, others: pl.DataFrame, pairs: pl.DataFrame,
          min_count: int = 2, min_purity: float = 0.5) -> dict:
    """pairs: (s1_id, match_id). Returns {"name": {...}, "addr": {...}}."""
    j = (
        pairs.join(others.select(pl.col("entity_id").alias("match_id"), pl.col("business_name").alias("qn"),
                                 pl.col("business_address").alias("qa")), on="match_id")
        .join(s1.select(pl.col("entity_id").alias("s1_id"), pl.col("business_name").alias("sn"),
                        pl.col("business_address").alias("sa")), on="s1_id")
    )
    # ---- names: positional alignment
    jn = j.filter(pl.col("qn").str.contains(NATIVE_RE)).select("qn", "sn")
    cnt: dict[str, Counter] = defaultdict(Counter)
    for qn, sn in jn.iter_rows():
        qt, st = qn.split(), sn.split()
        if len(qt) != len(st):
            continue
        for a, b in zip(qt, st):
            if any("ऀ" <= ch <= "෿" for ch in a):
                b2 = _latin(b)
                if b2:
                    cnt[_clean_native(a)][b2] += 1
    name_map = {}
    for a, c in cnt.items():
        b, n = c.most_common(1)[0]
        tot = sum(c.values())
        if n >= min_count and n / tot >= min_purity:
            name_map[a] = b
    # ---- addresses: native component -> most consistent S1 component
    ja = j.filter(pl.col("qa").str.contains(NATIVE_RE)).select("qa", "sa")
    comp_cnt: dict[str, Counter] = defaultdict(Counter)
    comp_tot: Counter = Counter()
    for qa, sa in ja.iter_rows():
        scomps = {c.strip() for c in sa.split(",") if c.strip()}
        for c in qa.split(","):
            c = _clean_native(c.strip())
            if c and any("ऀ" <= ch <= "෿" for ch in c):
                comp_tot[c] += 1
                for s in scomps:
                    comp_cnt[c][s] += 1
    addr_map = {}
    for c, cc in comp_cnt.items():
        s, n = cc.most_common(1)[0]
        if n >= min_count and n / comp_tot[c] >= 0.8:
            addr_map[c] = s
    return {"name": name_map, "addr": addr_map}


def save(maps: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(maps, f, ensure_ascii=False)


def load(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def apply_name_map(s: pl.Series, name_map: dict[str, str]) -> pl.Series:
    """Token-wise replacement of native-script name tokens (unique values only)."""
    mask = s.str.contains(NATIVE_RE)
    uniq = s.filter(mask).unique().to_list()
    if not uniq:
        return s
    conv = [" ".join(name_map.get(_clean_native(t), t) for t in x.split()) for x in uniq]
    m = pl.DataFrame({"_k": uniq, "_v": conv})
    out = pl.DataFrame({"_k": s}).with_row_index("_i").join(m, on="_k", how="left").sort("_i")
    return out.select(pl.coalesce("_v", "_k")).to_series().alias(s.name)


def apply_addr_map(s: pl.Series, addr_map: dict[str, str]) -> pl.Series:
    mask = s.str.contains(NATIVE_RE)
    uniq = s.filter(mask).unique().to_list()
    if not uniq:
        return s
    conv = [", ".join(addr_map.get(_clean_native(c.strip()), c.strip()) for c in x.split(",")) for x in uniq]
    m = pl.DataFrame({"_k": uniq, "_v": conv})
    out = pl.DataFrame({"_k": s}).with_row_index("_i").join(m, on="_k", how="left").sort("_i")
    return out.select(pl.coalesce("_v", "_k")).to_series().alias(s.name)

"""Vectorised (polars) normalisation of business names and addresses.

Everything here is rule-based and country-agnostic: the same rules run for every
country label, including ones never seen in training (France in the test set).
"""
from __future__ import annotations

import polars as pl
from anyascii import anyascii

from .translit import NATIVE_RE, apply_addr_map, apply_name_map

# --------------------------------------------------------------------------- names

# Markers that introduce an alternative (trade / former) name inside one field.
ALIAS_RE = (
    r"\b(?:d\s*/\s*b\s*/\s*a|dba|doing business as|trading as|t\s*/\s*a|formerly known as|"
    r"formerly|f\s*/\s*k\s*/\s*a|fka|a\s*/\s*k\s*/\s*a|aka)\b\s*:?|\*{2,}|\|"
)

NAME_CANON = {
    "private": "pvt", "pvt": "pvt", "prv": "pvt",
    "limited": "ltd", "ltd": "ltd", "limted": "ltd",
    "corporation": "corp", "corp": "corp",
    "company": "co", "co": "co", "cos": "co",
    "incorporated": "inc", "inc": "inc",
    "llc": "llc", "llp": "llp", "pllc": "pllc", "plc": "plc", "lp": "lp", "pc": "pc",
    "and": "and", "n": "and",
    "intl": "international", "int": "international",
    "mfg": "manufacturing", "svcs": "services", "svc": "services", "srvcs": "services",
    "tech": "technologies", "technology": "technologies",
    "assoc": "associates", "assocs": "associates",
    "bros": "brothers", "ent": "enterprises",
    "grp": "group", "ctr": "center", "centre": "center",
}

# Tokens that carry no identity: legal forms (incl. French forms, which only appear
# in test), honorifics, stop words.
NAME_STOP = {
    "pvt", "ltd", "corp", "co", "inc", "llc", "llp", "pllc", "plc", "lp", "pc", "public",
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "scop", "selarl", "gie", "eirl", "ei",
    "gmbh", "ag", "bv", "pte", "pty", "opc",
    "mr", "mrs", "ms", "dr", "shri", "sri", "smt", "m", "s", "messrs",
    "the", "and", "of", "et", "de", "du", "des", "la", "le", "les", "l", "d",
}

# --------------------------------------------------------------------------- addresses

ADDR_CANON = {
    "street": "st", "str": "st", "st": "st", "saint": "st", "sainte": "st",
    "road": "rd", "rd": "rd", "roda": "rd",
    "avenue": "ave", "ave": "ave", "av": "ave", "aven": "ave", "avn": "ave",
    "drive": "dr", "dr": "dr", "drv": "dr",
    "court": "ct", "ct": "ct", "crt": "ct",
    "lane": "ln", "ln": "ln",
    "place": "pl", "pl": "pl",
    "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "boul": "blvd", "bld": "blvd",
    "highway": "hwy", "hwy": "hwy", "hiway": "hwy",
    "parkway": "pkwy", "pkwy": "pkwy", "pky": "pkwy",
    "circle": "cir", "cir": "cir",
    "trail": "trl", "trl": "trl",
    "terrace": "ter", "ter": "ter",
    "square": "sq", "sq": "sq",
    "mount": "mt", "mt": "mt", "mountain": "mtn", "mtn": "mtn",
    "fort": "ft", "ft": "ft",
    "point": "pt", "pt": "pt",
    "heights": "hts", "hts": "hts",
    "expressway": "expy", "expy": "expy", "freeway": "fwy", "fwy": "fwy",
    "junction": "jct", "jct": "jct", "center": "ctr", "centre": "ctr", "ctr": "ctr",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "sector": "sector", "sec": "sector", "phase": "phase", "ph": "phase",
    "nagar": "nagar", "ngr": "nagar", "colony": "colony", "col": "colony",
    "marg": "marg", "mg": "marg",
    # French street types (test-only country): folded onto shared canonical forms
    "rue": "rue", "r": "rue", "chemin": "chemin", "ch": "chemin", "chem": "chemin",
    "impasse": "impasse", "imp": "impasse", "allee": "allee", "all": "allee",
    "route": "rte", "rte": "rte", "rt": "rte", "faubourg": "fbg", "fbg": "fbg",
    "cours": "cours", "crs": "cours", "quai": "quai",
}

ADDR_STOP = {
    "no", "h", "hn", "hno", "door", "plot", "flat", "house", "sno", "unit", "apt", "apartment",
    "ste", "suite", "fl", "flr", "floor", "bldg", "building", "po", "box", "pmb", "near", "nr",
    "opp", "opposite", "behind", "null", "none", "nan", "de", "du", "des", "la", "le", "les",
    "of", "the", "and", "dist", "district", "tq", "at", "post", "via", "room", "rm", "shop",
    "office", "number", "num", "bis", "et", "d", "l",
    "n", "s", "e", "w", "ne", "nw", "se", "sw",
}


def ascii_fold(s: pl.Series) -> pl.Series:
    """anyascii-transliterate only the non-ASCII values (done once per unique value)."""
    mask = s.str.contains(r"[^\x00-\x7F]")
    uniq = s.filter(mask).unique()
    if len(uniq) == 0:
        return s
    mapping = pl.DataFrame({"_k": uniq, "_v": [anyascii(x) for x in uniq.to_list()]})
    out = (
        pl.DataFrame({"_k": s})
        .with_row_index("_i")
        .join(mapping, on="_k", how="left")
        .sort("_i")
    )
    return out.select(pl.coalesce("_v", "_k")).to_series().alias(s.name)


def _tokens(expr: pl.Expr) -> pl.Expr:
    return expr.str.split(" ").list.eval(pl.element().filter(pl.element() != ""))


def _canon_list(expr: pl.Expr, mapping: dict[str, str]) -> pl.Expr:
    return expr.list.eval(pl.element().replace(mapping))


def _deleet_list(expr: pl.Expr) -> pl.Expr:
    """Undo digit-for-letter typos inside alphanumeric tokens: c0nsultants -> consultants."""
    e = pl.element()
    mixed = e.str.contains(r"[a-z]") & e.str.contains(r"\d") & ~e.str.contains(r"^\d+(?:st|nd|rd|th)$")
    fixed = e.str.replace_many(["0", "1", "3", "4", "5", "7", "@", "$"], ["o", "l", "e", "a", "s", "t", "a", "s"])
    return expr.list.eval(pl.when(mixed).then(fixed).otherwise(e))


def normalize_names(raw: pl.Series) -> pl.DataFrame:
    """Returns: nm (clean full name), nm_parts (alias cores), nm_tok (core tokens),
    nm_core (core string), nm_concat (core without spaces), nm_nonascii flag."""
    folded = ascii_fold(raw).str.to_lowercase()
    base = pl.DataFrame({"_s": folded, "_na": raw.str.contains(r"[^\x00-\x7F]")})
    s = pl.col("_s")
    s = (
        s.str.replace_all(r"\b\+?\d[\d\s\-]{6,}\d\b", " ")      # phone numbers
        .str.replace_all(r"\bwww\.", " ")
        .str.replace_all(r"\.(?:com|net|org|co\.in|in|fr|biz|info|co|us|io)\b", " ")
        .str.replace_all("&", " and ")
        .str.replace_all(r"['`’]", "")
        .str.replace_all(r"\.", "")
        .str.replace_all(ALIAS_RE, " | ")
        .str.replace_all(r"[^a-z0-9|]+", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )
    base = base.with_columns(s.alias("_c"))
    parts = (
        pl.col("_c").str.split("|")
        .list.eval(pl.element().str.strip_chars())
        .list.eval(pl.element().filter(pl.element() != ""))
    )
    base = base.with_columns(parts.alias("_parts"))
    # Per-part core strings (legal forms / honorifics removed).
    stop = list(NAME_STOP)
    part_core = pl.col("_parts").list.eval(
        pl.element()
        .str.split(" ")
        .list.eval(pl.element().replace(NAME_CANON))
        .list.eval(
            pl.when(
                pl.element().str.contains(r"[a-z]") & pl.element().str.contains(r"\d")
                & ~pl.element().str.contains(r"^\d+(?:st|nd|rd|th)$")
            )
            .then(pl.element().str.replace_many(["0", "1", "3", "4", "5", "7"], ["o", "l", "e", "a", "s", "t"]))
            .otherwise(pl.element())
        )
        .list.eval(pl.element().filter(~pl.element().is_in(stop) & (pl.element() != "")))
        .list.join(" ")
    )
    base = base.with_columns(part_core.alias("_pcore"))
    full_tok = _deleet_list(_canon_list(_tokens(pl.col("_c").str.replace_all(r"\|", " ")), NAME_CANON))
    base = base.with_columns(full_tok.alias("_ftok"))
    out = base.select(
        pl.col("_ftok").list.join(" ").alias("nm"),
        pl.col("_pcore").list.eval(pl.element().filter(pl.element() != "")).alias("nm_parts"),
        pl.col("_ftok").list.eval(pl.element().filter(~pl.element().is_in(stop))).list.unique(maintain_order=True).alias("nm_tok"),
        pl.col("_na").alias("nm_nonascii"),
    ).with_columns(
        pl.col("nm_tok").list.join(" ").alias("nm_core"),
        pl.col("nm_tok").list.join("").alias("nm_concat"),
    )
    return out


def normalize_addresses(raw: pl.Series) -> pl.DataFrame:
    """Returns: ad (clean address string), ad_comp (comma components), ad_tok (alpha tokens),
    ad_num (numeric tokens, leading zeros stripped), ad_first_num, ad_code (unit / plot codes
    such as "203-Q", "A3", "1-86", "58/3" with separators removed)."""
    folded = ascii_fold(raw).str.to_lowercase()
    base = pl.DataFrame({"_s": folded})
    e = pl.element()
    codes = (
        pl.col("_s").str.replace_all("#", " ")
        .str.extract_all(r"[a-z0-9]+(?:[\-/.][a-z0-9]+)*")
        .list.eval(e.filter(e.str.contains(r"\d")))
        .list.eval(e.str.replace_all(r"[\-/.]", "").str.replace(r"^0+", ""))
        .list.eval(e.filter(e.str.len_chars() > 1))
        .list.unique(maintain_order=True)
    )
    base = base.with_columns(codes.alias("_codes"))
    s = (
        pl.col("_s")
        .str.replace_all(r"\bnull\b", " ")
        .str.replace_all(r"(\d)([a-z])", "$1 $2")
        .str.replace_all(r"([a-z])(\d)", "$1 $2")
    )
    base = base.with_columns(s.alias("_s"))
    comps = (
        pl.col("_s").str.split(",")
        .list.eval(pl.element().str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars())
        .list.eval(pl.element().filter(pl.element() != ""))
    )
    toks = _tokens(pl.col("_s").str.replace_all(r"[^a-z0-9]+", " "))
    base = base.with_columns(comps.alias("_comps"), toks.alias("_toks"))
    stop = list(ADDR_STOP)
    out = base.select(
        pl.col("_toks").list.eval(e.str.replace(r"^0+(\d)", "$1")).list.eval(e.replace(ADDR_CANON)).list.join(" ").alias("ad"),
        pl.col("_comps").alias("ad_comp"),
        pl.col("_toks")
        .list.eval(e.filter(~e.str.contains(r"\d")))
        .list.eval(e.replace(ADDR_CANON))
        .list.eval(e.filter(~e.is_in(stop) & (e.str.len_chars() > 1)))
        .list.unique(maintain_order=True)
        .alias("ad_tok"),
        pl.col("_toks")
        .list.eval(e.filter(e.str.contains(r"^\d+$")).str.replace(r"^0+(\d)", "$1"))
        .alias("_nums"),
        pl.col("_codes").alias("ad_code"),
    ).with_columns(
        pl.col("_nums").list.first().alias("ad_first_num"),
        pl.col("_nums").list.unique(maintain_order=True).alias("ad_num"),
    ).drop("_nums")
    return out


def normalize(df: pl.DataFrame, maps: dict | None = None) -> pl.DataFrame:
    """Append normalised name/address columns. `maps` = learned transliteration maps."""
    names, addrs = df["business_name"], df["business_address"]
    native = names.str.contains(NATIVE_RE).alias("nm_native")
    if maps:
        names = apply_name_map(names, maps["name"])
        addrs = apply_addr_map(addrs, maps["addr"])
    return pl.concat([df, normalize_names(names), normalize_addresses(addrs), native.to_frame()], how="horizontal")

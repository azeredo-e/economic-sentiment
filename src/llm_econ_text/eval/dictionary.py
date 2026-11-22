"""Loughran-McDonald dictionary sentiment, reconstructing the thesis' TESI scoring (Almeida, 2024).

Per article: SS = sum(polarity_i * shifter_i) / n_words, polarity +1 (LM positive) / -1 (LM
negative), shifter from the PRECEDING word only. The thesis used sentometrics' English
valence-shifter list; R/sentometrics are not installed here, so the lists below are a documented
approximation (standard negators, a few amplifiers and de-amplifiers). The experiment about
stripping negators does not depend on their exact contents.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

LM_PATH = Path("data/raw/original/lm/Loughran_McDonald_en.xlsx")
STOPWORDS_PATH = Path(__file__).resolve().parents[1] / "data" / "nltk_english_stopwords.txt"

NEGATORS = {
    "not", "no", "nor", "never", "neither", "nobody", "nothing", "none", "nowhere", "without",
    "cannot", "can't", "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "won't", "wouldn't", "couldn't", "shouldn't", "hasn't", "haven't", "hadn't",
}
AMPLIFIERS = {"very", "extremely", "highly", "really", "most", "strongly", "sharply", "significantly"}
DEAMPLIFIERS = {"slightly", "barely", "hardly", "little", "less", "somewhat"}
_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(str(text).lower())


@lru_cache(maxsize=1)
def lexicon(path: Path = LM_PATH) -> dict[str, int]:
    d = pd.read_excel(path)
    pos = d.loc[d["sentiment"] == "positive", "token"].astype(str).str.lower()
    neg = d.loc[d["sentiment"] == "negative", "token"].astype(str).str.lower()
    return {**dict.fromkeys(pos, 1), **dict.fromkeys(neg, -1)}


@lru_cache(maxsize=1)
def stopwords() -> frozenset[str]:
    lines = STOPWORDS_PATH.read_text().splitlines()
    return frozenset(w for w in lines if w and not w.startswith("#"))


def score(text: str, shifter: bool = True, strip_stopwords: bool = False,
          lex: dict[str, int] | None = None) -> float:
    """Dictionary score of one text. `strip_stopwords` mimics the thesis' NLTK stop-word removal
    (which also removes negators and amplifiers) BEFORE scoring."""
    toks = tokenize(text)
    if strip_stopwords:
        sw = stopwords()
        toks = [t for t in toks if t not in sw]
    if not toks:
        return 0.0
    lex = lexicon() if lex is None else lex
    total = 0.0
    for i, t in enumerate(toks):
        pol = lex.get(t)
        if pol is None:
            continue
        mult = 1.0
        if shifter and i > 0:
            prev = toks[i - 1]
            if prev in NEGATORS:
                mult = -1.0
            elif prev in AMPLIFIERS:
                mult = 2.0
            elif prev in DEAMPLIFIERS:
                mult = 0.5
        total += pol * mult
    return total / len(toks)

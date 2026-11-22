"""Reconstruction of the thesis' "EPU article" rule, used only as an LLM-independent sampling
signal for the gold set (never as ground truth).

Derived from Almeida (2024), 03_Get_Loughran_McDonald.R / 05_Get_TEPU.R (MIT):
  - P terms: a hand-written list of generic political words;
  - U terms: the first 84 stems of the Loughran-McDonald *uncertainty* list (Porter2-stemmed);
  - an article is EPU if its translated tokens contain >= 1 P term AND >= 1 U term.
Quirk reproduced on purpose: the R code joins *unstemmed* tokens with *stemmed* U terms and
single-token P terms, so only words equal to their own stem match, and multi-word P terms
("central bank", "checks and balances") never match. The reconstruction is approximate: the
first-84 cut depends on the dictionary's row order.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import snowballstemmer

TERMS_PATH = Path(__file__).with_name("thesis_epu_terms.json")

P_TERMS = [
    "Accountability", "Ambassador", "Autocracy", "Ballot", "Bipartisanship", "Bureaucracy",
    "Campaign", "Campaigning", "Cabinet", "Checks and balances", "Chancellor", "Central Bank",
    "Civil servant", "Civil service", "Coalition", "Congress", "Congressman", "Constituency",
    "Constituent", "Constitution", "Corruption", "Democracy", "Dictatorship", "Diplomat",
    "Diplomacy", "Domestic policy", "Electoral", "Electoral college", "Election", "Executive",
    "Federalism", "Foreign policy", "Government", "Governing party", "Gridlock",
    "House of Representatives", "Human rights", "Impeachment", "Initiative",
    "International relations", "Legislation", "Legislative", "Lobby", "Lobbyist", "Lawmaker",
    "Manifesto", "Majority", "Minority", "Monarch", "Monarchy", "Oligarchy", "Opposition",
    "Opposition party", "Parliament", "Petition", "Policy", "Political campaign",
    "Political party", "Politics", "Poll", "President", "President-elect", "Public servant",
    "Recall", "Recall election", "Referendum", "Republic", "Ruling party", "Senate",
    "Separation of powers", "Suffrage", "Transparency", "Vote",
]


def build_terms(lm_xlsx: str | Path) -> dict[str, list[str]]:
    """Rebuild the term lists from the original repo's Loughran_McDonald_en.xlsx."""
    import pandas as pd

    lm = pd.read_excel(lm_xlsx)
    words = lm.loc[lm["sentiment"] == "uncertainty", "token"].astype(str).str.lower()
    stem = snowballstemmer.stemmer("english").stemWord
    stems: list[str] = []
    for w in words:
        s = stem(w)
        if s not in stems:
            stems.append(s)
    return {"P": sorted({t.lower() for t in P_TERMS}), "U": stems[:84]}


@lru_cache(maxsize=1)
def _sets() -> tuple[frozenset[str], frozenset[str]]:
    terms = json.loads(TERMS_PATH.read_text())
    # only single-token P terms can ever match an unstemmed word token (see docstring)
    p = frozenset(t for t in terms["P"] if " " not in t and "-" not in t)
    return p, frozenset(terms["U"])


def thesis_epu_flag(translated_text: str) -> bool:
    p, u = _sets()
    toks = set(re.findall(r"[a-z]+", str(translated_text).lower()))
    return bool(toks & p) and bool(toks & u)

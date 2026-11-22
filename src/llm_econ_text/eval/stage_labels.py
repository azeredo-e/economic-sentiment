"""Append hand-entered labels (chunk by chunk) to a staging CSV and rebuild the review workbook.

Provenance: these are CLAUDE's labels (made blind to the model predictions, from the labelling
guide), written to data/gold/claude_labels.csv and shown in data/gold/labels_claude.xlsx for a
human to judge. The staging CSV is the pristine record; edits go in the workbook.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_econ_text.eval.gold_labels import load_gold
from llm_econ_text.eval.make_label_sheet import build

STAGING = Path("data/gold/claude_labels.csv")
WORKBOOK = Path("data/gold/labels_claude.xlsx")
COLS = ["item", "relevant", "topic", "sentiment", "epu", "ambiguous", "note", "confidence"]


def append(rows: list[tuple]) -> None:
    new = pd.DataFrame(rows, columns=COLS)
    old = (pd.read_csv(STAGING, dtype=str, keep_default_na=False)
           if STAGING.exists() else pd.DataFrame(columns=COLS))
    if dup := set(new["item"]) & set(old["item"]):
        raise ValueError(f"already labelled: {sorted(dup)}")
    pd.concat([old, new], ignore_index=True).to_csv(STAGING, index=False)


def _human_edited(staged: pd.DataFrame) -> bool:
    """True if the workbook's labels differ from the staged (pristine) ones, i.e. someone edited it."""
    if not WORKBOOK.exists():
        return False
    wb, _, _ = load_gold(str(WORKBOOK))
    pr, _, _ = load_gold(str(STAGING))
    if wb.empty:
        return False
    m = wb.merge(pr, on="item", suffixes=("_wb", "_pr"))
    cols = ["g_relevant", "g_topic", "g_sentiment", "g_epu", "ambiguous"]

    def norm(col: pd.Series) -> pd.Series:  # None and NaN both mean "blank"
        return col.map(lambda v: "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))

    return len(m) != len(pr) or any((norm(m[f"{c}_wb"]) != norm(m[f"{c}_pr"])).any() for c in cols)


def rebuild(force: bool = False) -> None:
    staged = pd.read_csv(STAGING, dtype=str, keep_default_na=False)
    if not force and _human_edited(staged):
        raise RuntimeError(f"{WORKBOOK} contains edits that differ from {STAGING}; "
                           "refusing to overwrite (pass force=True to discard them)")
    build(Path("data/gold/to_label.csv"), WORKBOOK, prefill=staged)
    valid, problems, pending = load_gold(str(WORKBOOK))
    print(f"labelled {len(staged)} | parser: valid {len(valid)}, problems {len(problems)}, "
          f"pending {pending}")
    for p in problems:
        print("  PROBLEM:", p)

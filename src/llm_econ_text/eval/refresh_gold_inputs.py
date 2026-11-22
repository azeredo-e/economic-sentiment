"""Rebuild the headline + lead text of the EXISTING gold sample from the current pool.

Used after fixing the lead extractor: same 300 articles, same item ids and order, new input text.
Backs up the old blind file first. Rebuild the workbook afterwards with make_label_sheet --force.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from llm_econ_text.data.sampling import POOL_PATH
from llm_econ_text.eval.draw_gold import LABEL_COLUMNS


def refresh(design_path: str = "data/gold/gold_design.csv",
            blind_path: str = "data/gold/to_label.csv", backup_suffix: str = "_old_leads") -> pd.DataFrame:
    design = pd.read_csv(design_path)
    # keep_default_na=False: an article whose body is empty (only metadata) has lead == "",
    # which is valid, not "missing"
    pool = pd.read_csv(POOL_PATH, keep_default_na=False)[["article_id", "headline", "lead"]]
    merged = design[["item", "article_id"]].merge(pool, on="article_id", how="left", indicator=True)
    if (merged["_merge"] != "both").any():
        raise ValueError("some gold articles are missing from the pool")
    merged = merged.drop(columns="_merge")
    blind = Path(blind_path)
    if blind.exists():
        shutil.copy(blind, blind.with_name(blind.stem + backup_suffix + blind.suffix))
    out = merged[["item", "headline", "lead"]].copy()
    for c in LABEL_COLUMNS:
        out[c] = ""
    out.to_csv(blind, index=False, encoding="utf-8-sig")
    return out


def main() -> None:
    out = refresh()
    print(f"rewrote {len(out)} items")


if __name__ == "__main__":
    main()

"""Stratified pool of raw articles and the thesis-style EPU flags used for the gold-set draw."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_econ_text.data.loaders import TOPIC_GLOBS, load_topic
from llm_econ_text.data.thesis_epu import thesis_epu_flag

CLEANED_DIR = Path("data/raw/original/3_textos_tratados")
STUDY_END = pd.Timestamp("2023-12-31")
POOL_PATH = Path("data/processed/pool.csv")
FLAGS_PATH = Path("data/processed/thesis_flags.csv")


def build_pool(per_cell: int = 70, seed: int = 42) -> pd.DataFrame:
    """Sample up to `per_cell` raw articles per (source topic x year).

    Adds `cell_total` (corpus size of the cell) and `cell_n` (pool size), so that
    `w_pool = cell_total / cell_n` weights the pool back to the full raw corpus.
    """
    parts = []
    for topic in TOPIC_GLOBS:
        df = load_topic(topic)
        df = df[df["date"] <= STUDY_END].copy()  # thesis window: May 2010 - Dec 2023
        df["year"] = df["date"].dt.year
        for _, g in df.groupby("year"):
            take = g.sample(min(len(g), per_cell), random_state=seed).copy()
            take["cell_total"], take["cell_n"] = len(g), len(take)
            parts.append(take)
    pool = pd.concat(parts, ignore_index=True)
    pool["w_pool"] = pool["cell_total"] / pool["cell_n"]
    return pool


def build_thesis_flags() -> pd.DataFrame:
    """url -> thesis_hit, for the articles that survived the thesis' own cleaning."""
    rows = []
    for f in sorted(CLEANED_DIR.glob("*.xlsx")):
        df = pd.read_excel(f, usecols=["endereco", "Traducao"])
        rows.append(pd.DataFrame({
            "url": df["endereco"],
            "thesis_hit": [thesis_epu_flag(t) for t in df["Traducao"].fillna("")],
        }))
    return pd.concat(rows, ignore_index=True).drop_duplicates("url")


def cached(path: Path, build) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = build()
    df.to_csv(path, index=False)
    return df

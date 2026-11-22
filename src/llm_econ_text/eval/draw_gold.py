"""Draw the 300-article gold sample from the scored pool (no API calls).

Strata (mutually exclusive), using the current prompt's labels and the thesis-style EPU flag:
  S2 llm_epu          LLM relevant and EPU-flagged                       (up to 40, all if fewer)
  S4 thesis_hit       LLM relevant, not EPU, thesis P&U keyword hit      (50, EPU-recall check)
  S3 llm_irrelevant   LLM says not relevant                              (60, missed-relevant check)
  S1 llm_relevant     LLM relevant, not EPU, no thesis hit               (the remainder)
Inside a stratum the draw round-robins over topic x year cells. Each article gets a weight
w = w_pool * N_cell / n_cell (back to the full raw corpus), then calibrated so each stratum's
weights sum to the population it represents (cells left undrawn would otherwise be lost).

Writes data/gold/to_label.csv (blind: opaque ids, headline + lead only) and
data/processed/gold_design.csv (strata, weights, split, model labels; do NOT open while labelling).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from llm_econ_text.data.sampling import FLAGS_PATH, POOL_PATH

LABEL_COLUMNS = ["relevant", "topic", "sentiment", "epu", "ambiguous", "note"]


def round_robin_take(df: pd.DataFrame, n: int, cell_cols: list[str], seed: int) -> pd.DataFrame:
    """Take n rows spreading evenly over cells (one per cell per pass)."""
    if len(df) <= n:
        return df.copy()
    rng = np.random.default_rng(seed)
    d = df.assign(_r=rng.random(len(df)))
    d["_rank"] = d.groupby(cell_cols)["_r"].rank(method="first")
    d = d.assign(_tie=rng.random(len(d))).sort_values(["_rank", "_tie"])
    return d.head(n).drop(columns=["_r", "_rank", "_tie"])


def assign_strata(pool: pd.DataFrame) -> pd.DataFrame:
    p = pool.copy()
    p["stratum"] = np.select(
        [~p.relevant, p.epu, p.thesis_hit],
        ["S3_llm_irrelevant", "S2_llm_epu", "S4_thesis_hit"],
        default="S1_llm_relevant",
    )
    p["strat_topic"] = np.where(p.relevant, p.llm_topic, p.source_topic)
    return p


def calibrate(gold: pd.DataFrame, assigned_pool: pd.DataFrame) -> pd.DataFrame:
    """Post-stratify: scale weights so each stratum's weights sum to the corpus population that
    stratum represents (sum of w_pool). The round-robin draw leaves some topic x year cells with
    no draws, so the raw cell weights under-cover the population (measured 0.71 and 0.87 for two
    strata on the real draw); calibration removes that to first order."""
    pop = assigned_pool.groupby("stratum").w_pool.sum()
    g = gold.copy()
    g["stratum_pop_weight"] = g["stratum"].map(pop)
    g["weight"] = (
        g["weight_raw"] * g["stratum_pop_weight"] / g.groupby("stratum")["weight_raw"].transform("sum")
    )
    return g


def draw(pool: pd.DataFrame, total: int = 300, seed: int = 42,
         quotas: dict[str, int] | None = None) -> pd.DataFrame:
    quotas = quotas or {"S2_llm_epu": 40, "S4_thesis_hit": 50, "S3_llm_irrelevant": 60}
    p = assign_strata(pool)
    cell = ["strat_topic", "year"]
    taken = []
    for s, q in quotas.items():
        taken.append(round_robin_take(p[p.stratum == s], q, cell, seed))
    already = sum(len(t) for t in taken)
    taken.append(round_robin_take(p[p.stratum == "S1_llm_relevant"], total - already, cell, seed))
    gold = pd.concat(taken, ignore_index=True)

    pop_cell = p.groupby(["stratum", *cell]).size().rename("N_cell")
    got = gold.groupby(["stratum", *cell]).size().rename("n_cell_gold")
    gold = gold.join(pop_cell, on=["stratum", *cell]).join(got, on=["stratum", *cell])
    gold["weight_raw"] = gold["w_pool"] * gold["N_cell"] / gold["n_cell_gold"]
    gold = calibrate(gold, p)

    # dev/test: one third dev, balanced inside each stratum
    rng = np.random.default_rng(seed + 1)
    gold["split"] = "test"
    for idx in gold.groupby("stratum").groups.values():
        idx = list(idx)
        rng.shuffle(idx)
        gold.loc[idx[: round(len(idx) / 3)], "split"] = "dev"
    gold = gold.sample(frac=1, random_state=seed).reset_index(drop=True)
    gold["item"] = [f"g{i + 1:03d}" for i in range(len(gold))]
    return gold


def scored_pool(scorer) -> pd.DataFrame:
    """Pool articles joined with thesis flags and the current model labels (cache only)."""
    pool = pd.read_csv(POOL_PATH).merge(pd.read_csv(FLAGS_PATH), on="url", how="left")
    pool["thesis_hit"] = pool["thesis_hit"].fillna(False).astype(bool)
    recs = [scorer.get_label(r.article_id, r.headline, r.lead) for r in pool.itertuples()]
    pool = pool[[x is not None for x in recs]].copy()
    labs = [x[0] for x in recs if x is not None]
    pool["relevant"] = [x.relevant for x in labs]
    pool["llm_topic"] = [x.topic for x in labs]
    pool["llm_sentiment"] = [x.sentiment for x in labs]
    pool["epu"] = [x.epu for x in labs]
    pool["llm_confidence"] = [x.confidence for x in labs]
    return pool


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv

    from llm_econ_text.eval.score_pool import make_scorer

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--total", type=int, default=300)
    ap.add_argument("--recalibrate", action="store_true",
                    help="re-weight the EXISTING design file (no new draw)")
    args = ap.parse_args(argv)
    load_dotenv()
    cfg = yaml.safe_load(Path(args.config).read_text())
    scorer = make_scorer(cfg)
    pool = scored_pool(scorer)
    design_path = Path("data/processed/gold_design.csv")

    if args.recalibrate:
        d = pd.read_csv(design_path)
        if "weight_raw" not in d:
            d["weight_raw"] = d["weight"]
        d = calibrate(d, assign_strata(pool))
        d.to_csv(design_path, index=False)
        print("recalibrated:", design_path)
        print(d.groupby("stratum").agg(pop=("stratum_pop_weight", "first"),
                                       sum_weights=("weight", "sum")).round(1).to_string())
        return

    print(f"scored pool: {len(pool)} | strata sizes:\n"
          f"{assign_strata(pool).stratum.value_counts().to_string()}")
    gold = draw(pool, total=args.total, seed=cfg["sampling"]["seed"])
    Path("data/gold").mkdir(parents=True, exist_ok=True)
    blind = gold[["item", "headline", "lead"]].copy()
    for c in LABEL_COLUMNS:
        blind[c] = ""
    blind.to_csv("data/gold/to_label.csv", index=False, encoding="utf-8-sig")
    keep = ["item", "article_id", "url", "date", "source_topic", "stratum", "strat_topic",
            "weight", "weight_raw", "stratum_pop_weight", "w_pool", "N_cell", "n_cell_gold",
            "split", "relevant", "llm_topic", "llm_sentiment", "epu", "llm_confidence",
            "thesis_hit"]
    gold[keep].to_csv(design_path, index=False)
    print(gold.groupby(["stratum", "split"]).size().unstack().to_string())
    print("by LLM topic (relevant only):",
          gold[gold.relevant].llm_topic.value_counts().to_dict())


if __name__ == "__main__":
    main()

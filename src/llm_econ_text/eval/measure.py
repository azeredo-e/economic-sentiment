"""Measure real per-article cost/latency on a small stratified sample, then project the full run.

Usage: python -m llm_econ_text.eval.measure --per-topic 10
Spends real API money (logged in the SQLite ledger and bounded by the budget guard).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from llm_econ_text.data.loaders import TOPIC_GLOBS, load_topic
from llm_econ_text.llm.client import BudgetExceeded, Scorer


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-topic", type=int, default=10)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args(argv)

    load_dotenv()
    cfg = yaml.safe_load(Path(args.config).read_text())
    version = cfg["prompt_version"]
    system = Path(f"prompts/{version}/system.md").read_text()
    scorer = Scorer(
        system, version,
        model=cfg["llm"]["bulk_model"], effort=cfg["llm"]["effort"],
        db_path=cfg["cache"]["path"],
        total_usd=cfg["budget"]["total_usd"],
        hard_stop_fraction=cfg["budget"]["hard_stop_fraction"],
    )

    frames = []
    for topic in TOPIC_GLOBS:
        df = load_topic(topic)
        frames.append(df.sample(args.per_topic, random_state=cfg["sampling"]["seed"]))
    sample = pd.concat(frames, ignore_index=True)

    rows = []
    spent0 = scorer.total_spent()
    for r in sample.itertuples():
        try:
            res = scorer.score(r.article_id, r.headline, r.lead)
        except BudgetExceeded as e:
            print("STOP:", e)
            break
        lab = res.label
        rows.append({
            "article_id": r.article_id, "source_topic": r.source_topic, "date": r.date,
            "status": res.status, "cost": res.cost_usd,
            "topic": lab.topic if lab else None, "sentiment": lab.sentiment if lab else None,
            "epu": lab.epu if lab else None, "confidence": lab.confidence if lab else None,
            "headline": r.headline[:70],
        })
    out = pd.DataFrame(rows)
    out_path = Path(f"data/processed/measure_{args.per_topic}pt.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    placeholders = ",".join("?" * len(out))
    calls = pd.read_sql_query(
        f"SELECT * FROM calls WHERE article_id IN ({placeholders}) AND prompt_hash=?",
        scorer.db, params=[*out["article_id"], scorer.prompt_hash],
    )
    paid = calls[calls["cost_usd"] > 0]
    print(out.drop(columns=["date"]).to_string(index=False))
    print(f"\nstatus counts: {out['status'].value_counts().to_dict()}")
    print(f"spent this run: ${scorer.total_spent() - spent0:.4f} | ledger total: "
          f"${scorer.total_spent():.4f} of ${cfg['budget']['total_usd']:.2f}")
    if len(paid):
        per = paid["cost_usd"].mean()
        print(f"billed calls: {len(paid)} | mean in/cache_w/cache_r/out tokens: "
              f"{paid['input_tokens'].mean():.0f}/{paid['cache_write_tokens'].mean():.0f}/"
              f"{paid['cache_read_tokens'].mean():.0f}/{paid['output_tokens'].mean():.0f}"
              f" | mean latency {paid['latency_s'].mean():.1f}s")
        print(f"mean cost/article ${per:.5f} -> 25k articles ${per * 25_000:.2f}, "
              f"80k articles ${per * 80_000:.2f} (no batch discount)")


if __name__ == "__main__":
    main()

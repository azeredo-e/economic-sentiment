"""Score the stratified pool with the current prompt through the Batch API.

  python -m llm_econ_text.eval.score_pool --pilot 20     # small batch first
  python -m llm_econ_text.eval.score_pool                # whole pool
The batch id is saved to data/processed/batch_ids.txt the moment it is submitted; if the process
dies, re-run with --collect <batch_id>. Spends real API money (guarded by the budget).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from llm_econ_text.data.sampling import POOL_PATH, build_pool, cached
from llm_econ_text.llm.client import Scorer


def make_scorer(cfg: dict) -> Scorer:
    version = cfg["prompt_version"]
    return Scorer(
        Path(f"prompts/{version}/system.md").read_text(), version,
        model=cfg["llm"]["bulk_model"], effort=cfg["llm"]["effort"],
        db_path=cfg["cache"]["path"], total_usd=cfg["budget"]["total_usd"],
        hard_stop_fraction=cfg["budget"]["hard_stop_fraction"],
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0, help="score only N pool articles")
    ap.add_argument("--collect", default=None, help="collect an already-submitted batch id")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args(argv)

    load_dotenv()
    cfg = yaml.safe_load(Path(args.config).read_text())
    scorer = make_scorer(cfg)
    spent0 = scorer.total_spent()

    batch_id = args.collect
    if batch_id is None:
        pool = cached(POOL_PATH, lambda: build_pool(seed=cfg["sampling"]["seed"]))
        if args.pilot:
            pool = pool.sample(args.pilot, random_state=cfg["sampling"]["seed"])
        items = [(r.article_id, r.headline, r.lead) for r in pool.itertuples()]
        print(f"pool articles: {len(items)}")
        batch_id = scorer.submit_batch(items)
        if batch_id is None:
            print("everything already scored (cache)")
            return
        with open("data/processed/batch_ids.txt", "a") as fh:
            fh.write(f"{batch_id}\tprompt={scorer.prompt_version}\tn={len(items)}\n")
        print("submitted batch", batch_id, "- waiting (usually < 1 hour)")
    b = scorer.wait_batch(batch_id)
    print("batch ended:", b.request_counts)
    print("collected:", scorer.collect_batch(batch_id))
    print(f"spent this run: ${scorer.total_spent() - spent0:.4f} | ledger total: "
          f"${scorer.total_spent():.4f}")

    pool = pd.read_csv(POOL_PATH, keep_default_na=False)
    rows = []
    for r in pool.itertuples():
        got = scorer.get_label(r.article_id, r.headline, r.lead)
        if got:
            lab, coerced = got
            rows.append((lab.relevant, lab.topic, lab.epu, coerced))
    df = pd.DataFrame(rows, columns=["relevant", "topic", "epu", "coerced"])
    print(f"scored pool articles: {len(df)} | relevant {df.relevant.mean():.1%} | "
          f"EPU flags {int(df.epu.sum())} | coerced {int(df.coerced.sum())}")


if __name__ == "__main__":
    main()

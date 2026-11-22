"""Translate the gold articles' headline + lead into English (M4).

  python -m llm_econ_text.eval.translate_gold
Writes data/gold/translations_haiku_v1.csv with columns item, headline, lead (English), so it
can be scored with `predict_gold --inputs`. The file holds translated article text, so it is
gitignored. Spends real API money (about $0.05, cached).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from llm_econ_text.llm.client import BudgetExceeded
from llm_econ_text.llm.translate import Translator


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="data/gold/translations_haiku_v1.csv")
    args = ap.parse_args(argv)
    load_dotenv()
    cfg = yaml.safe_load(Path(args.config).read_text())
    tr = Translator(
        Path("prompts/translate_v1.md").read_text(), "translate_v1",
        model=cfg["llm"]["bulk_model"], effort=cfg["llm"]["effort"], db_path=cfg["cache"]["path"],
        total_usd=cfg["budget"]["total_usd"], hard_stop_fraction=cfg["budget"]["hard_stop_fraction"])
    design = pd.read_csv("data/gold/gold_design.csv")[["item", "article_id"]]
    blind = pd.read_csv("data/gold/to_label.csv", dtype=str, keep_default_na=False)
    items = design.merge(blind[["item", "headline", "lead"]], on="item")
    spent0, rows, failed = tr.s.total_spent(), [], []
    for r in items.itertuples():
        try:
            t = tr.translate(r.article_id, r.headline, r.lead)
        except BudgetExceeded as e:
            print("STOP:", e)
            break
        if t is None:
            failed.append(r.item)
            continue
        rows.append({"item": r.item, "headline": t.headline_en, "lead": t.lead_en})
    pd.DataFrame(rows).to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"translated {len(rows)} of {len(items)} | failed {failed} | spent "
          f"${tr.s.total_spent() - spent0:.4f} (ledger ${tr.s.total_spent():.4f})")


if __name__ == "__main__":
    main()

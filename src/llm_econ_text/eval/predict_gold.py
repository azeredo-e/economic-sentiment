"""Score the gold articles with the current prompt/model on the CURRENT lead text.

  python -m llm_econ_text.eval.predict_gold --name haiku_v2
Writes results/preds_<name>.csv (item, relevant, topic, sentiment, epu, confidence) for
`evaluate --pred`. Spends real API money (cached per article id + input text + prompt).
A rare output whose sentiment_label contradicts its score is repaired from the score (the label
is redundant) instead of being dropped, which would silently shrink the evaluation set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from llm_econ_text.eval.score_pool import make_scorer
from llm_econ_text.llm.client import BudgetExceeded
from llm_econ_text.llm.schema import ArticleLabel, label_from_score, normalize_label


def repaired_label(scorer, article_id: str) -> ArticleLabel | None:
    row = scorer.db.execute(
        "SELECT label_json FROM calls WHERE article_id=? AND status='invalid' "
        "ORDER BY id DESC LIMIT 1", (article_id,)).fetchone()
    try:
        raw = json.loads(row[0])
        raw["sentiment_label"] = label_from_score(float(raw["sentiment"]))
        return ArticleLabel.model_validate(raw)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):  # ValidationError is a ValueError
        return None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="haiku_v2")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results")
    ap.add_argument("--inputs", default="data/gold/to_label.csv",
                    help="csv with item, headline, lead (e.g. the English translations)")
    ap.add_argument("--id-suffix", default="",
                    help="appended to the cache id, e.g. '#en' or '#pt2' for a retest control")
    args = ap.parse_args(argv)
    load_dotenv()
    cfg = yaml.safe_load(Path(args.config).read_text())
    scorer = make_scorer(cfg)
    design = pd.read_csv("data/gold/gold_design.csv")[["item", "article_id"]]
    blind = pd.read_csv(args.inputs, dtype=str, keep_default_na=False)
    items = design.merge(blind[["item", "headline", "lead"]], on="item")
    spent0 = scorer.total_spent()
    rows, repaired, missing = [], 0, []
    for r in items.itertuples():
        cid = r.article_id + args.id_suffix
        try:
            res = scorer.score(cid, r.headline, r.lead)
        except BudgetExceeded as e:
            print("STOP:", e)
            break
        label = res.label
        if label is None:
            label = repaired_label(scorer, cid)
            repaired += label is not None
        if label is None:
            missing.append(r.item)
            continue
        label, _ = normalize_label(label)
        rows.append({"item": r.item, "relevant": label.relevant, "topic": label.topic,
                     "sentiment": label.sentiment, "epu": label.epu,
                     "confidence": label.confidence})
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"preds_{args.name}.csv", index=False)
    print(f"predictions: {len(rows)} of {len(items)} | repaired labels: {repaired} | "
          f"missing: {missing} | spent ${scorer.total_spent() - spent0:.4f} "
          f"(ledger ${scorer.total_spent():.4f})")


if __name__ == "__main__":
    main()

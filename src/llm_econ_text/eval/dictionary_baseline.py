"""Dictionary baseline (the thesis' method) vs the accepted gold labels and vs the LLM.

  python -m llm_econ_text.eval.dictionary_baseline
Arms (sentiment, on gold-relevant items that the LLM also called relevant, so both methods are
scored on the same articles):
  thesis_text  LM dictionary + shifter on the thesis' OWN cleaned English text (full article,
               stop words stripped) - only for the gold items that survived its cleaning
  en_plain     LM dictionary, no shifter, on our faithful English translation of headline + lead
  en_shifter   same text, with the negation shifter
  en_stripped  same text, NLTK stop words removed first (mimics the thesis), then the shifter
The 3-class neutral band is tuned on the dev split only and reported on the test split; the
threshold-free Spearman is also given. Also writes the thesis pipeline's predictions for all four
tasks (relevance = survived its keyword filter, topic = its search folder, EPU = its P&U rule).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from llm_econ_text.data.sampling import CLEANED_DIR
from llm_econ_text.eval import dictionary as dic
from llm_econ_text.eval.gold_labels import SENTIMENTS, load_gold
from llm_econ_text.eval.metrics import multiclass_report
from llm_econ_text.eval.translation_effect import load_pred

ORD = {"negative": -1, "neutral": 0, "positive": 1}
GRID = [0, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04]


def label_of(score: float, tau: float) -> str:
    return "positive" if score > tau else "negative" if score < -tau else "neutral"


def thesis_texts(urls: set[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for f in sorted(CLEANED_DIR.glob("*.xlsx")):
        df = pd.read_excel(f, usecols=["endereco", "Traducao"])
        for u, t in zip(df["endereco"], df["Traducao"], strict=True):
            if u in urls:
                out[u] = str(t)
    return out


def tune_and_test(df: pd.DataFrame, col: str) -> dict:
    """df: gold-relevant items with columns gold, weight, split and `col` (the score)."""
    dev, test = df[df.split == "dev"], df[df.split == "test"]
    best = max(GRID, key=lambda t: multiclass_report(
        dev.gold, dev[col].map(lambda s: label_of(s, t)), SENTIMENTS, dev.weight)["macro_f1"] or 0)

    def metrics(part: pd.DataFrame, tau: float) -> dict:
        r = multiclass_report(part.gold, part[col].map(lambda s: label_of(s, tau)), SENTIMENTS,
                              part.weight)
        return {"n": len(part), "macro_f1": r["macro_f1"], "kappa": r["kappa"],
                "acc_unweighted": float((part.gold == part[col].map(lambda s: label_of(s, tau))).mean())}

    rho = lambda part: float(spearmanr(part[col], part.gold.map(ORD)).statistic)
    rng = np.random.default_rng(0)
    boots = [rho(df.iloc[rng.integers(0, len(df), len(df))]) for _ in range(500)]
    return {"tau_tuned_on_dev": best, "test_at_tuned_tau": metrics(test, best),
            "all_at_tau0": metrics(df, 0.0), "spearman_all": rho(df),
            "spearman_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/gold/claude_labels.csv")
    ap.add_argument("--design", default="data/gold/gold_design.csv")
    ap.add_argument("--pt", default="results/preds_haiku_v2_newleads.csv")
    ap.add_argument("--en-text", default="data/gold/translations_haiku_v1.csv")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    gold, _, _ = load_gold(args.gold)
    design = pd.read_csv(args.design)
    pt = load_pred(args.pt)
    en = pd.read_csv(args.en_text, keep_default_na=False).set_index("item")
    d = gold.merge(design[["item", "url", "weight", "split", "source_topic", "thesis_hit"]], on="item")
    d["llm_relevant"] = d["item"].map(pt["relevant"])
    d["llm_score"] = d["item"].map(pt["sentiment"])
    d["en_text"] = d["item"].map(lambda i: f"{en.loc[i, 'headline']} {en.loc[i, 'lead']}")
    tt = thesis_texts(set(d.url))
    d["thesis_text"] = d["url"].map(tt)
    d["in_cleaned"] = d["thesis_text"].notna()

    d["s_en_plain"] = d.en_text.map(lambda t: dic.score(t, shifter=False))
    d["s_en_shifter"] = d.en_text.map(lambda t: dic.score(t, shifter=True))
    d["s_en_stripped"] = d.en_text.map(lambda t: dic.score(t, shifter=True, strip_stopwords=True))
    d["s_thesis_text"] = d.thesis_text.map(lambda t: dic.score(t, shifter=True) if isinstance(t, str) else np.nan)

    rel = d[d.g_relevant & d.llm_relevant].rename(columns={"g_sentiment": "gold"}).copy()
    res: dict = {"n_gold_relevant_and_llm_relevant": len(rel), "arms": {}}
    for arm in ["en_plain", "en_shifter", "en_stripped"]:
        res["arms"][arm] = tune_and_test(rel, f"s_{arm}")
    sub = rel[rel.in_cleaned]
    res["arms"]["thesis_text"] = {**tune_and_test(sub, "s_thesis_text"), "n_items_in_thesis_cleaned_set": len(sub)}
    llm = rel.assign(llm=rel.llm_score)
    res["llm_same_items"] = tune_and_test(llm.assign(), "llm")
    res["llm_on_thesis_subset"] = tune_and_test(llm[llm.in_cleaned], "llm")
    # how often does each dictionary arm find any sentiment word at all (zero score = neutral by construction)
    res["share_zero_score"] = {a: float((rel[f"s_{a}"] == 0).mean()) for a in ["en_plain", "en_shifter", "en_stripped"]}

    # thesis pipeline predictions for all four tasks, for `evaluate --pred`
    tau = res["arms"]["thesis_text"]["tau_tuned_on_dev"]
    pipe = pd.DataFrame({
        "item": d["item"], "relevant": d["in_cleaned"],
        "topic": np.where(d["in_cleaned"], d["source_topic"], "none"),
        "sentiment": [0.0 if not ok else (0.5 if s > tau else -0.5 if s < -tau else 0.0)
                      for ok, s in zip(d["in_cleaned"], d["s_thesis_text"].fillna(0), strict=True)],
        "epu": d["in_cleaned"] & d["thesis_hit"].astype(bool)})
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    pipe.to_csv(out / "preds_thesis_pipeline.csv", index=False)
    (out / "dictionary_baseline.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

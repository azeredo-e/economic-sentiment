"""M4: how much does PT -> EN translation change the LLM's labels, beyond sampling noise?

  python -m llm_econ_text.eval.translation_effect
Arms (same prompt, same model): PT = Portuguese scored once; PT2 = the same Portuguese scored a
second time (noise control: temperature cannot be set, so repeated scoring differs); EN = the
English translation. Translation effect = flip(PT, EN) - flip(PT, PT2). 'Robust flips' are items
where PT and PT2 agree but EN differs: the ones attributable to translation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from llm_econ_text.eval.evaluate import build_frame, section
from llm_econ_text.eval.gold_labels import load_gold
from llm_econ_text.llm.schema import label_from_score

KINDS = ["relevance", "topic", "sentiment", "epu"]


def load_pred(path: str) -> pd.DataFrame:
    d = pd.read_csv(path).set_index("item")
    d["label"] = d["sentiment"].map(label_from_score)
    d.loc[~d["relevant"], ["topic", "label"]] = ["none", "neutral"]
    return d


def flip_flags(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """One row per item; NaN where the kind is undefined (topic/sentiment need both relevant)."""
    both = a["relevant"] & b["relevant"]
    f = pd.DataFrame(index=a.index)
    f["relevance"] = (a["relevant"] != b["relevant"]).astype(float)
    f["topic"] = np.where(both, (a["topic"] != b["topic"]).astype(float), np.nan)
    f["sentiment"] = np.where(both, (a["label"] != b["label"]).astype(float), np.nan)
    f["epu"] = (a["epu"] != b["epu"]).astype(float)
    return f


def rates(f: pd.DataFrame) -> dict:
    return {k: float(np.nanmean(f[k])) for k in KINDS}


def boot(fa: pd.DataFrame, fb: pd.DataFrame, n: int = 1000, seed: int = 0) -> dict:
    """95% CIs for flip rate of each pair and of their difference, resampling items."""
    rng = np.random.default_rng(seed)
    out = {k: {"a": [], "b": [], "diff": []} for k in KINDS}
    idx = np.arange(len(fa))
    for _ in range(n):
        s = rng.choice(idx, len(idx))
        for k in KINDS:
            x, y = fa[k].to_numpy()[s], fb[k].to_numpy()[s]
            if np.isnan(x).all() or np.isnan(y).all():
                continue
            out[k]["a"].append(np.nanmean(x))
            out[k]["b"].append(np.nanmean(y))
            out[k]["diff"].append(np.nanmean(y) - np.nanmean(x))
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
    return {k: {m: ci(v) for m, v in d.items()} for k, d in out.items()}


def score_shift(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    both = a["relevant"] & b["relevant"]
    x, y = a.loc[both, "sentiment"].to_numpy(), b.loc[both, "sentiment"].to_numpy()
    d = y - x
    return {"n": int(both.sum()), "mean_signed_shift": round(float(d.mean()), 3),
            "mean_abs_shift": round(float(np.abs(d).mean()), 3),
            "share_abs_shift_ge_0.2": round(float((np.abs(d) >= 0.2).mean()), 3),
            "spearman": round(float(spearmanr(x, y).statistic), 3)}


def gold_agreement(gold, design, path) -> dict:
    df = build_frame(gold, design, pd.read_csv(path))
    s = section(df, n_boot=0)
    return {"relevance_macro_f1": s["relevance"]["weighted"]["macro_f1"],
            "sentiment_macro_f1": s["sentiment"]["weighted"]["macro_f1"],
            "sentiment_acc_unweighted": s["sentiment"]["unweighted"]["accuracy"],
            "epu_f1": s["epu"]["weighted"]["f1"]}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="results/preds_haiku_v2_newleads.csv")
    ap.add_argument("--pt2", default="results/preds_haiku_v2_pt_retest.csv")
    ap.add_argument("--en", default="results/preds_haiku_v2_en.csv")
    ap.add_argument("--gold", default="data/gold/claude_labels.csv")
    ap.add_argument("--design", default="data/gold/gold_design.csv")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    pt, pt2, en = load_pred(args.pt), load_pred(args.pt2), load_pred(args.en)
    items = pt.index.intersection(pt2.index).intersection(en.index)
    pt, pt2, en = pt.loc[items], pt2.loc[items], en.loc[items]

    f_noise, f_tr, f_tr2 = flip_flags(pt, pt2), flip_flags(pt, en), flip_flags(pt2, en)
    res = {
        "n_items": len(items),
        "flip_rate": {"noise_PT_vs_PT2": rates(f_noise), "translation_PT_vs_EN": rates(f_tr),
                      "translation_PT2_vs_EN": rates(f_tr2)},
        "ci95": boot(f_noise, f_tr),
        "score_shift": {"noise_PT_vs_PT2": score_shift(pt, pt2),
                        "translation_PT_vs_EN": score_shift(pt, en)},
    }
    gold, _, _ = load_gold(args.gold)
    design = pd.read_csv(args.design)
    res["agreement_with_gold"] = {n: gold_agreement(gold, design, p) for n, p in
                                  (("PT", args.pt), ("PT2", args.pt2), ("EN", args.en))}

    stable = (pt["relevant"] == pt2["relevant"]) & (pt["label"] == pt2["label"]) \
        & (pt["topic"] == pt2["topic"]) & (pt["epu"] == pt2["epu"])
    changed = (pt["relevant"] != en["relevant"]) | (pt["label"] != en["label"]) \
        | (pt["topic"] != en["topic"]) | (pt["epu"] != en["epu"])
    robust = items[stable & changed]
    res["items_all_four_labels_identical_PT_vs_PT2"] = int(stable.sum())
    res["robust_flip_items"] = len(robust)
    flips = pd.DataFrame({"item": robust,
                          "pt_relevant": pt.loc[robust, "relevant"].to_numpy(),
                          "en_relevant": en.loc[robust, "relevant"].to_numpy(),
                          "pt_topic": pt.loc[robust, "topic"].to_numpy(),
                          "en_topic": en.loc[robust, "topic"].to_numpy(),
                          "pt_sentiment": pt.loc[robust, "sentiment"].to_numpy(),
                          "en_sentiment": en.loc[robust, "sentiment"].to_numpy(),
                          "pt_epu": pt.loc[robust, "epu"].to_numpy(),
                          "en_epu": en.loc[robust, "epu"].to_numpy()})
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    flips.to_csv(out / "translation_robust_flips.csv", index=False)
    (out / "translation_effect.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

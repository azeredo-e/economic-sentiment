"""Evaluate a method's predictions against the hand-labelled gold set.

  python -m llm_econ_text.eval.evaluate --name haiku_v2 --pred results/preds_haiku_v2_newleads.csv
  python -m llm_econ_text.eval.evaluate --name lm_dict --pred results/preds_lm_dict.csv
  python -m llm_econ_text.eval.evaluate --name llm_v2 --relabel data/gold/relabel.csv

Predictions come from --pred (e.g. results/preds_haiku_v2_newleads.csv). Another method
supplies a CSV with columns: item, relevant (bool), topic, sentiment (float -1..1), epu (bool).
Works on a partly filled labels file (only fully valid rows are scored). Makes no API calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from llm_econ_text.eval.gold_labels import SENTIMENTS, TOPICS, load_gold
from llm_econ_text.eval.metrics import (
    binary_report,
    confusion,
    multiclass_report,
    spearman,
    stratified_ci,
)
from llm_econ_text.llm.schema import label_from_score

_ORD = {"negative": -1, "neutral": 0, "positive": 1}


def build_frame(gold: pd.DataFrame, design: pd.DataFrame, preds: pd.DataFrame | None):
    d = design.copy()
    if preds is None:
        if "llm_topic" not in d.columns:
            raise SystemExit("no predictions: pass --pred results/preds_<name>.csv "
                             "(the committed design file carries none)")
        p = d[["item", "relevant", "llm_topic", "llm_sentiment", "epu"]]
        p.columns = ["item", "p_relevant", "p_topic", "p_score", "p_epu"]
    else:
        p = preds[["item", "relevant", "topic", "sentiment", "epu"]].copy()
        p.columns = ["item", "p_relevant", "p_topic", "p_score", "p_epu"]
    df = gold.merge(d[["item", "stratum", "weight", "split", "source_topic", "thesis_hit"]],
                    on="item").merge(p, on="item")
    df["p_relevant"] = df["p_relevant"].astype(bool)
    df["p_epu"] = df["p_epu"].astype(bool) & df["p_relevant"]
    df["p_score"] = np.where(df["p_relevant"], df["p_score"], 0.0)
    df["p_topic"] = np.where(df["p_relevant"], df["p_topic"], "none")
    df["p_sentiment"] = df["p_score"].map(label_from_score)
    return df


def _ci(df, fn, n_boot):
    return stratified_ci(df, fn, n_boot=n_boot)


def section(df: pd.DataFrame, n_boot: int) -> dict:
    w = df["weight"]
    out: dict = {"n_labelled": len(df), "kish_n_eff": float(w.sum() ** 2 / (w**2).sum())}
    rel = df[df.g_relevant]

    out["relevance"] = {
        "weighted": binary_report(df.g_relevant, df.p_relevant, w),
        "unweighted": binary_report(df.g_relevant, df.p_relevant),
        "ci_macro_f1": _ci(df, lambda b: binary_report(
            b.g_relevant, b.p_relevant, b.weight)["macro_f1"], n_boot),
        "confusion": confusion(df.g_relevant.map({True: "relevant", False: "irrelevant"}),
                               df.p_relevant.map({True: "relevant", False: "irrelevant"}),
                               ["relevant", "irrelevant"]),
    }
    if len(rel):
        out["topic"] = {
            "weighted": multiclass_report(rel.g_topic, rel.p_topic, [*TOPICS, "none"],
                                          rel.weight, f1_labels=TOPICS),
            "unweighted": multiclass_report(rel.g_topic, rel.p_topic, [*TOPICS, "none"],
                                            f1_labels=TOPICS),
            "confusion": confusion(rel.g_topic, rel.p_topic, [*TOPICS, "none"]),
        }
        sent_fn = lambda b: multiclass_report(
            b.g_sentiment, b.p_sentiment, SENTIMENTS, b.weight)["macro_f1"]
        out["sentiment"] = {
            "weighted": multiclass_report(rel.g_sentiment, rel.p_sentiment, SENTIMENTS,
                                          rel.weight, ordinal=True),
            "unweighted": multiclass_report(rel.g_sentiment, rel.p_sentiment, SENTIMENTS,
                                            ordinal=True),
            "spearman_score_vs_gold": spearman(rel.g_sentiment.map(_ORD).to_numpy(),
                                               rel.p_score.to_numpy()),
            "ci_macro_f1": _ci(rel, sent_fn, n_boot),
            "confusion": confusion(rel.g_sentiment, rel.p_sentiment, SENTIMENTS),
            "per_topic": {
                t: {"n": len(g), "accuracy": multiclass_report(
                    g.g_sentiment, g.p_sentiment, SENTIMENTS, g.weight)["accuracy"]}
                for t, g in rel.groupby("g_topic")
            },
        }
        calm = rel[~rel.ambiguous]
        out["sentiment"]["macro_f1_excl_ambiguous"] = multiclass_report(
            calm.g_sentiment, calm.p_sentiment, SENTIMENTS, calm.weight)["macro_f1"]
    out["epu"] = {
        "weighted": binary_report(df.g_epu, df.p_epu, w),
        "unweighted": binary_report(df.g_epu, df.p_epu),
        "ci_f1": _ci(df, lambda b: binary_report(b.g_epu, b.p_epu, b.weight)["f1"], n_boot),
        "confusion": confusion(df.g_epu.map({True: "epu", False: "not"}),
                               df.p_epu.map({True: "epu", False: "not"}), ["epu", "not"]),
    }
    return out


def strata_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("stratum")
    return pd.DataFrame({
        "n": g.size(),
        "gold_relevant": g.g_relevant.mean().round(2),
        "llm_relevant": g.p_relevant.mean().round(2),
        "gold_epu": g.g_epu.mean().round(2),
        "llm_epu": g.p_epu.mean().round(2),
    })


def intra_annotator(first: pd.DataFrame, second: pd.DataFrame) -> dict:
    from sklearn.metrics import cohen_kappa_score

    m = first.merge(second, on="item", suffixes=("_1", "_2"))
    out = {"n_items": len(m)}
    for col in ["g_relevant", "g_topic", "g_sentiment", "g_epu"]:
        sub = m.dropna(subset=[f"{col}_1", f"{col}_2"])
        out[col] = (float(cohen_kappa_score(sub[f"{col}_1"], sub[f"{col}_2"]))
                    if len(sub) > 2 and sub[f"{col}_1"].nunique() > 1 else None)
    return out


def _f(v, d=3):
    return "n/a" if v is None else f"{v:.{d}f}"


def _ci_s(ci):
    return "" if ci is None else f" (95% CI {ci[0]:.2f}-{ci[1]:.2f})"


def _row(*parts: str) -> str:
    return "".join(parts)


def render(name: str, meta: dict, res: dict, strata: pd.DataFrame, warnings: list[str],
           intra: dict | None) -> str:
    L = [f"# Evaluation: {name}", "",
         _row(f"gold labelled: {meta['n_labelled']} of {meta['n_total']} ",
              f"(pending {meta['pending']}, rejected {meta['problems']})"), ""]
    for w in warnings:
        L.append(f"> WARNING: {w}")
    if warnings:
        L.append("")
    for split in ("test", "dev", "all"):
        s = res.get(split)
        if not s or not s["n_labelled"]:
            continue
        L += [_row(f"## Split: {split} (n={s['n_labelled']}, ",
                   f"effective n for weighted estimates ~{s['kish_n_eff']:.0f})"), ""]
        r, rw, ru = s["relevance"], s["relevance"]["weighted"], s["relevance"]["unweighted"]
        L += ["### Relevance",
              _row(f"weighted: acc {_f(rw['accuracy'])} | P {_f(rw['precision'])} | ",
                   f"R {_f(rw['recall'])} | F1 {_f(rw['f1'])} | macro-F1 {_f(rw['macro_f1'])}",
                   _ci_s(r["ci_macro_f1"]), f" | kappa {_f(rw['kappa'])}"),
              _row(f"unweighted: acc {_f(ru['accuracy'])} | macro-F1 {_f(ru['macro_f1'])} | ",
                   f"kappa {_f(ru['kappa'])}"),
              "```", r["confusion"].to_string(), "```", ""]
        if "topic" in s:
            t, se = s["topic"], s["sentiment"]
            tw, sw, su = t["weighted"], se["weighted"], se["unweighted"]
            per_topic = ", ".join(f"{k} {_f(v['accuracy'], 2)} (n={v['n']})"
                                  for k, v in se["per_topic"].items())
            L += ["### Topic (gold-relevant articles)",
                  _row(f"weighted: acc {_f(tw['accuracy'])} | macro-F1 {_f(tw['macro_f1'])} | ",
                       f"kappa {_f(tw['kappa'])}"),
                  "```", t["confusion"].to_string(), "```", "",
                  "### Sentiment, 3-class (gold-relevant articles)",
                  _row(f"weighted: acc {_f(sw['accuracy'])} | macro-F1 {_f(sw['macro_f1'])}",
                       _ci_s(se["ci_macro_f1"]), f" | kappa {_f(sw['kappa'])} | ",
                       f"linear-kappa {_f(sw['kappa_linear'])}"),
                  _row(f"unweighted: acc {_f(su['accuracy'])} | macro-F1 {_f(su['macro_f1'])} | ",
                       f"Spearman (score vs gold) {_f(se['spearman_score_vs_gold'])} | ",
                       f"macro-F1 excl. ambiguous {_f(se['macro_f1_excl_ambiguous'])}"),
                  "```", se["confusion"].to_string(), "```",
                  f"per topic (weighted accuracy): {per_topic}", ""]
        e = s["epu"]
        ew, eu = e["weighted"], e["unweighted"]
        L += ["### EPU (all articles)",
              f"gold positives {ew['n_pos_true']} | predicted positives {ew['n_pos_pred']}",
              _row(f"weighted: P {_f(ew['precision'])} | R {_f(ew['recall'])} | ",
                   f"F1 {_f(ew['f1'])}", _ci_s(e["ci_f1"]), f" | kappa {_f(ew['kappa'])}"),
              _row(f"unweighted: P {_f(eu['precision'])} | R {_f(eu['recall'])} | ",
                   f"F1 {_f(eu['f1'])}"),
              "```", e["confusion"].to_string(), "```", ""]
    L += ["## Strata diagnostics (all labelled)", "```", strata.to_string(), "```",
          _row("Read: S3 `gold_relevant` is the share of articles the LLM called irrelevant ",
               "that you judged relevant (missed relevant). S4 `gold_epu` checks whether ",
               "thesis-keyword hits the LLM did not flag are really EPU (LLM recall)."), ""]
    if intra:
        L += ["## Intra-annotator agreement (Cohen's kappa, first pass vs re-label)",
              json.dumps(intra, indent=1), ""]
    return "\n".join(L)


def _jsonable(o):
    if isinstance(o, pd.DataFrame):
        return o.to_dict()
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, tuple):
        return list(o)
    return o


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/gold/claude_labels.csv",
                    help="accepted gold labels (csv or the reviewed xlsx)")
    ap.add_argument("--design", default="data/gold/gold_design.csv")
    ap.add_argument("--pred", default=None, help="CSV: item, relevant, topic, sentiment, epu")
    ap.add_argument("--relabel", default=None, help="second-pass labels for intra-annotator kappa")
    ap.add_argument("--name", default="llm")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args(argv)

    gold, problems, pending = load_gold(args.gold)
    design = pd.read_csv(args.design)
    preds = pd.read_csv(args.pred) if args.pred else None
    df = build_frame(gold, design, preds)

    warnings = []
    if problems:
        warnings.append(f"{len(problems)} label rows rejected: " + "; ".join(problems[:8])
                        + (" ..." if len(problems) > 8 else ""))
    if len(df) < len(design):
        warnings.append(f"only {len(df)} of {len(design)} items labelled: estimates use the "
                        "design weights of a full sample, so treat them as provisional")
    if df.g_epu.sum() < 20:
        warnings.append(f"only {int(df.g_epu.sum())} gold EPU positives: EPU metrics are very "
                        "noisy; read the strata table and the counts, not the F1")

    res = {"test": section(df[df.split == "test"], args.n_boot),
           "dev": section(df[df.split == "dev"], args.n_boot),
           "all": section(df, args.n_boot)} if len(df) else {}
    intra = None
    if args.relabel:
        second, _, _ = load_gold(args.relabel)
        intra = intra_annotator(gold, second)

    meta = {"n_labelled": len(df), "n_total": len(design), "pending": pending,
            "problems": len(problems)}
    md = render(args.name, meta, res, strata_table(df) if len(df) else pd.DataFrame(),
                warnings, intra) if len(df) else "# No valid labelled rows yet\n" + "\n".join(problems)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / f"eval_{args.name}.md").write_text(md)
    (out / f"eval_{args.name}.json").write_text(
        json.dumps(_jsonable({"meta": meta, "results": res, "intra": intra}), indent=1,
                   default=str))
    print(md)


if __name__ == "__main__":
    main()

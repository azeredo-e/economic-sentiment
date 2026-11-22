"""How much did the human reviewer change Claude's labels?

  python -m llm_econ_text.eval.agreement
Compares the pristine labels (data/gold/claude_labels.csv) with the reviewed workbook
(data/gold/labels_claude.xlsx, edited in place). Reports, per field, how many items were
overruled and Cohen's kappa between the two labelings. Items still blank in the reviewed file
are skipped. This is the evidence for how far Claude's labels can stand in for human gold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from llm_econ_text.eval.gold_labels import load_gold

FIELDS = ["g_relevant", "g_topic", "g_sentiment", "g_epu"]


def compare(pristine: str, reviewed: str) -> dict:
    a, _, _ = load_gold(pristine)
    b, problems, pending = load_gold(reviewed)
    m = a.merge(b, on="item", suffixes=("_claude", "_human"))
    out: dict = {"items_compared": len(m), "reviewed_file_problems": problems,
                 "reviewed_file_pending": pending, "fields": {}}
    changed_any = pd.Series(False, index=m.index)
    for f in FIELDS:
        x, y = m[f"{f}_claude"].fillna(""), m[f"{f}_human"].fillna("")
        diff = x.astype(str) != y.astype(str)
        changed_any |= diff
        both = pd.concat([x, y], axis=1).astype(str)
        kappa = (float(cohen_kappa_score(both.iloc[:, 0], both.iloc[:, 1]))
                 if len(m) > 2 and both.iloc[:, 0].nunique() > 1 else None)
        out["fields"][f] = {"changed": int(diff.sum()),
                            "pct_changed": round(float(diff.mean()) * 100, 1) if len(m) else None,
                            "kappa": None if kappa is None else round(kappa, 3),
                            "changed_items": list(m.loc[diff, "item"])}
    out["items_with_any_change"] = int(changed_any.sum())
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pristine", default="data/gold/claude_labels.csv")
    ap.add_argument("--reviewed", default="data/gold/labels_claude.xlsx")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    res = compare(args.pristine, args.reviewed)
    Path(args.out).mkdir(exist_ok=True)
    (Path(args.out) / "agreement_claude_vs_reviewer.json").write_text(json.dumps(res, indent=1))
    print(f"items compared: {res['items_compared']} | items with any change: "
          f"{res['items_with_any_change']}")
    for f, r in res["fields"].items():
        print(f"  {f:13s} overruled {r['changed']:3d} ({r['pct_changed']}%) | kappa {r['kappa']}")


if __name__ == "__main__":
    main()

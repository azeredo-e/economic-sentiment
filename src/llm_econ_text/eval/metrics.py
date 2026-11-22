"""Classification metrics with sampling weights and stratified bootstrap intervals.

The gold set is a stratified, EPU-oversampled sample, so headline numbers use the per-article
`weight` (inverse inclusion probability back to the raw corpus). Unweighted numbers are kept
alongside because they describe the labelled sample itself.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)


def _safe(fn, *a, **k):
    try:
        v = float(fn(*a, **k))
        return None if np.isnan(v) else v
    except ValueError:
        return None


def binary_report(y_true, y_pred, w=None) -> dict:
    """Positive class = True."""
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    kw = {"sample_weight": w}
    return {
        "n": len(y_true), "n_pos_true": int(y_true.sum()), "n_pos_pred": int(y_pred.sum()),
        "accuracy": _safe(accuracy_score, y_true, y_pred, **kw),
        "precision": _safe(precision_score, y_true, y_pred, zero_division=0, **kw),
        "recall": _safe(recall_score, y_true, y_pred, zero_division=0, **kw),
        "f1": _safe(f1_score, y_true, y_pred, zero_division=0, **kw),
        "macro_f1": _safe(f1_score, y_true, y_pred, average="macro", labels=[False, True],
                          zero_division=0, **kw),
        "kappa": _safe(cohen_kappa_score, y_true, y_pred, labels=[False, True], **kw),
    }


def multiclass_report(y_true, y_pred, labels: list[str], w=None, ordinal=False,
                      f1_labels: list[str] | None = None) -> dict:
    """`labels` is the full label set (used for kappa, so a prediction outside the real classes,
    e.g. topic 'none', still counts as a disagreement). `f1_labels` restricts the macro average
    to the real classes."""
    kw = {"sample_weight": w}
    out = {
        "n": len(y_true),
        "accuracy": _safe(accuracy_score, y_true, y_pred, **kw),
        "macro_f1": _safe(f1_score, y_true, y_pred, average="macro",
                          labels=f1_labels or labels, zero_division=0, **kw),
        "kappa": _safe(cohen_kappa_score, y_true, y_pred, labels=labels, **kw),
    }
    if ordinal:
        out["kappa_linear"] = _safe(cohen_kappa_score, y_true, y_pred, labels=labels,
                                    weights="linear", **kw)
    return out


def confusion(y_true, y_pred, labels: list[str]) -> pd.DataFrame:
    """Unweighted counts; rows = gold, columns = predicted."""
    return pd.crosstab(pd.Categorical(y_true, labels), pd.Categorical(y_pred, labels),
                       rownames=["gold"], colnames=["pred"], dropna=False)


def spearman(a, b) -> float | None:
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return _safe(lambda: spearmanr(a, b).statistic)


def stratified_ci(df: pd.DataFrame, fn: Callable[[pd.DataFrame], float | None],
                  n_boot: int = 500, seed: int = 0, strata: str = "stratum"):
    """95% bootstrap interval, resampling articles within each sampling stratum."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby(strata)]
    vals = []
    for _ in range(n_boot):
        boot = pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in groups])
        v = fn(boot)
        if v is not None:
            vals.append(v)
    if len(vals) < max(20, n_boot // 5):
        return None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

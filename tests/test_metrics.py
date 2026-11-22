import numpy as np
import pandas as pd
import pytest

from llm_econ_text.eval.metrics import (
    binary_report,
    confusion,
    multiclass_report,
    spearman,
    stratified_ci,
)


def test_binary_perfect_and_known_case():
    r = binary_report([1, 1, 0, 0], [1, 1, 0, 0])
    assert r["f1"] == 1 and r["kappa"] == 1
    r = binary_report([1, 1, 0, 0], [1, 0, 0, 0])  # TP1 FN1 TN2
    assert r["precision"] == 1 and r["recall"] == 0.5 and r["accuracy"] == 0.75


def test_weights_change_the_estimate():
    y, p = [1, 0, 0, 0], [1, 1, 0, 0]
    unweighted = binary_report(y, p)["precision"]
    weighted = binary_report(y, p, [1, 10, 1, 1])["precision"]  # the false positive counts 10x
    assert weighted < unweighted


def test_multiclass_and_confusion():
    labs = ["negative", "neutral", "positive"]
    t = ["negative", "neutral", "positive", "positive"]
    p = ["negative", "neutral", "neutral", "positive"]
    r = multiclass_report(t, p, labs, ordinal=True)
    assert r["accuracy"] == 0.75 and r["kappa_linear"] is not None
    c = confusion(t, p, labs)
    assert c.loc["positive", "neutral"] == 1 and c.values.sum() == 4


def test_spearman_handles_constant_input():
    assert spearman(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])) is None
    assert spearman(np.array([-1, 0, 1]), np.array([-0.5, 0.0, 0.7])) == pytest.approx(1.0)


def test_stratified_ci_brackets_the_estimate():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"stratum": rng.choice(["a", "b"], n), "y": rng.random(n) < 0.5})
    df["p"] = df.y ^ (rng.random(n) < 0.2)  # ~80% accurate
    df["weight"] = 1.0
    fn = lambda b: binary_report(b.y, b.p, b.weight)["accuracy"]
    lo, hi = stratified_ci(df, fn, n_boot=200)
    assert lo < fn(df) < hi and hi - lo < 0.2


def test_topic_kappa_counts_none_predictions_as_errors():
    topics = ["industria", "servicos"]
    t, p = ["industria", "servicos", "servicos"], ["industria", "none", "servicos"]
    r = multiclass_report(t, p, [*topics, "none"], f1_labels=topics)
    assert r["accuracy"] < 1 and r["kappa"] < 1
    assert multiclass_report(t, t, [*topics, "none"], f1_labels=topics)["kappa"] == 1

import pandas as pd

from llm_econ_text.eval.translation_effect import flip_flags, rates, score_shift


def preds(rows):
    return pd.DataFrame(rows, columns=["item", "relevant", "topic", "sentiment", "epu"]).set_index("item")


def test_flip_flags_only_count_topic_and_sentiment_when_both_relevant():
    a = preds([("g1", True, "industria", -0.5, False), ("g2", True, "servicos", 0.5, False),
               ("g3", False, "none", 0.0, False)])
    b = preds([("g1", True, "industria", 0.5, False), ("g2", False, "none", 0.0, False),
               ("g3", False, "none", 0.0, False)])
    for d in (a, b):
        d["label"] = d["sentiment"].map(lambda s: "negative" if s <= -0.2 else "positive" if s >= 0.2 else "neutral")
    f = flip_flags(a, b)
    assert list(f["relevance"]) == [0, 1, 0]
    assert f["sentiment"].iloc[0] == 1 and pd.isna(f["sentiment"].iloc[1])
    assert rates(f)["relevance"] == 1 / 3 and rates(f)["sentiment"] == 1.0


def test_score_shift_signed_and_abs():
    a = preds([("g1", True, "x", 0.0, False), ("g2", True, "x", 0.5, False)])
    b = preds([("g1", True, "x", 0.3, False), ("g2", True, "x", 0.2, False)])
    s = score_shift(a, b)
    assert s["mean_signed_shift"] == 0.0 and s["mean_abs_shift"] == 0.3
    assert s["share_abs_shift_ge_0.2"] == 1.0

import pandas as pd

from llm_econ_text.eval.agreement import compare

COLS = ["item", "headline", "lead", "relevant", "topic", "sentiment", "epu", "ambiguous", "note"]


def write(path, rows):
    pd.DataFrame([{**dict.fromkeys(COLS, ""), **r} for r in rows]).to_csv(path, index=False)


def test_counts_overruled_items_per_field(tmp_path):
    base = [
        {"item": "g1", "relevant": "y", "topic": "industria", "sentiment": "negative", "epu": "n"},
        {"item": "g2", "relevant": "y", "topic": "servicos", "sentiment": "positive", "epu": "n"},
        {"item": "g3", "relevant": "n"},
        {"item": "g4", "relevant": "y", "topic": "agronegocio", "sentiment": "neutral", "epu": "y"},
    ]
    edited = [dict(r) for r in base]
    edited[1]["sentiment"] = "neutral"  # reviewer overrules g2 sentiment
    edited[2] = {"item": "g3", "relevant": "y", "topic": "servicos", "sentiment": "neutral", "epu": "n"}
    write(tmp_path / "a.csv", base)
    write(tmp_path / "b.csv", edited)
    r = compare(str(tmp_path / "a.csv"), str(tmp_path / "b.csv"))
    assert r["items_with_any_change"] == 2
    assert r["fields"]["g_sentiment"]["changed_items"] == ["g2", "g3"]
    assert r["fields"]["g_relevant"]["changed_items"] == ["g3"]
    assert r["fields"]["g_epu"]["changed"] == 0

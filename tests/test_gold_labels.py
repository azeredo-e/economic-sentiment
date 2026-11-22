import pandas as pd

from llm_econ_text.eval.gold_labels import load_gold


def write(tmp_path, rows):
    cols = ["item", "headline", "lead", "relevant", "topic", "sentiment", "epu", "ambiguous", "note"]
    p = tmp_path / "labels.csv"
    pd.DataFrame([{**dict.fromkeys(cols, ""), **r} for r in rows]).to_csv(p, index=False)
    return str(p)


def test_friendly_spellings_and_pending(tmp_path):
    p = write(tmp_path, [
        {"item": "g1", "relevant": "sim", "topic": "Indústria", "sentiment": "Positivo", "epu": "n"},
        {"item": "g2", "relevant": "N"},
        {"item": "g3"},  # not labelled yet
        {"item": "g4", "relevant": "y", "topic": "mercado financeiro", "sentiment": "neg",
         "epu": "s", "ambiguous": "y"},
    ])
    df, problems, pending = load_gold(p)
    assert problems == [] and pending == 1
    r = df.set_index("item")
    assert r.loc["g1", "g_topic"] == "industria" and r.loc["g1", "g_sentiment"] == "positive"
    assert r.loc["g2", "g_topic"] == "none" and r.loc["g2", "g_epu"] is False or not r.loc["g2", "g_epu"]
    assert r.loc["g4", "g_topic"] == "mercado_financeiro" and r.loc["g4", "g_epu"]
    assert r.loc["g4", "ambiguous"]


def test_problems_name_the_item(tmp_path):
    p = write(tmp_path, [
        {"item": "g1", "relevant": "maybe"},
        {"item": "g2", "relevant": "y", "topic": "esportes", "sentiment": "neutral", "epu": "n"},
        {"item": "g3", "relevant": "y", "topic": "servicos", "sentiment": "meh", "epu": "n"},
        {"item": "g4", "relevant": "n", "epu": "y"},
        {"item": "g5", "relevant": "y", "topic": "servicos", "sentiment": "neutral", "epu": ""},
    ])
    df, problems, _ = load_gold(p)
    assert df.empty and len(problems) == 5
    assert all(f"g{i}" in problems[i - 1] for i in range(1, 6))


def test_label_workbook_roundtrip_and_overwrite_protection(tmp_path):
    import pytest
    from openpyxl import load_workbook

    from llm_econ_text.eval.make_label_sheet import main

    src = tmp_path / "to_label.csv"
    pd.DataFrame({"item": ["g001", "g002"], "headline": ["h1", "h2"], "lead": ["l1", "l2"]}
                 ).to_csv(src, index=False)
    out = tmp_path / "labels.xlsx"
    main(["--src", str(src), "--out", str(out)])
    df, problems, pending = load_gold(str(out))
    assert df.empty and problems == [] and pending == 2  # nothing labelled yet

    wb = load_workbook(out)
    ws = wb["Labels"]
    ws["D2"], ws["E2"], ws["F2"], ws["G2"] = "y", "servicos", "positive", "n"
    wb.save(out)
    df, problems, pending = load_gold(str(out))
    assert problems == [] and pending == 1 and df.iloc[0].g_topic == "servicos"

    with pytest.raises(SystemExit):  # must not clobber an existing workbook
        main(["--src", str(src), "--out", str(out)])

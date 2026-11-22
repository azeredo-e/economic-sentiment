import pandas as pd
from openpyxl import load_workbook

from llm_econ_text.eval import stage_labels as sl


def setup(tmp_path, monkeypatch):
    blind = tmp_path / "to_label.csv"
    pd.DataFrame({"item": ["g001", "g002"], "headline": ["h1", "h2"], "lead": ["l1", "l2"]}
                 ).to_csv(blind, index=False)
    monkeypatch.setattr(sl, "STAGING", tmp_path / "claude_labels.csv")
    monkeypatch.setattr(sl, "WORKBOOK", tmp_path / "labels_claude.xlsx")
    sl.append([
        ("g001", "y", "servicos", "positive", "n", "n", "x", "high"),
        ("g002", "n", "", "", "", "n", "irrelevant row has blank topic/sentiment", "medium"),
    ])
    staged = pd.read_csv(sl.STAGING, dtype=str, keep_default_na=False)
    sl.build(blind, sl.WORKBOOK, prefill=staged)  # same builder rebuild() uses
    return staged


def test_guard_ignores_an_untouched_workbook_even_with_blank_rows(tmp_path, monkeypatch):
    staged = setup(tmp_path, monkeypatch)
    assert sl._human_edited(staged) is False


def test_guard_detects_a_real_edit(tmp_path, monkeypatch):
    staged = setup(tmp_path, monkeypatch)
    wb = load_workbook(sl.WORKBOOK)
    wb["Labels"]["F2"] = "negative"  # a reviewer overrules the sentiment of g001
    wb.save(sl.WORKBOOK)
    assert sl._human_edited(staged) is True

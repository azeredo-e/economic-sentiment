"""Parse and validate the hand-filled gold label file (data/gold/labels.csv).

Accepts friendly spellings (y/n/s/sim/nao, 'Indústria', 'mercado financeiro', 'neg'/'positivo').
Rows that are blank (not labelled yet) are reported as pending; rows with bad or inconsistent
values are reported as problems with the item id. Only fully valid rows are returned, so the
file can be evaluated while it is still being filled in.
"""

from __future__ import annotations

import unicodedata

import pandas as pd

TOPICS = ["industria", "agronegocio", "mercado_financeiro", "mercado_trabalho", "servicos"]
SENTIMENTS = ["negative", "neutral", "positive"]
_YES = {"y", "yes", "s", "sim", "true", "1", "x"}
_NO = {"n", "no", "nao", "false", "0"}
_SENT = {
    "negative": "negative", "neg": "negative", "negativo": "negative", "-1": "negative",
    "neutral": "neutral", "neu": "neutral", "neutro": "neutral", "0": "neutral",
    "positive": "positive", "pos": "positive", "positivo": "positive", "1": "positive",
}
_TOPIC_ALIASES = {
    "agro": "agronegocio", "industrial": "industria", "financeiro": "mercado_financeiro",
    "trabalho": "mercado_trabalho", "servico": "servicos",
}


def _norm(v: object) -> str:
    s = unicodedata.normalize("NFKD", str(v)).encode("ascii", "ignore").decode().strip().lower()
    return s.replace(" ", "_") if s else ""


def _bool(v: object) -> bool | None:
    s = _norm(v)
    return True if s in _YES else False if s in _NO else None


def _topic(v: object) -> str | None:
    s = _norm(v)
    s = _TOPIC_ALIASES.get(s, s)
    return s if s in TOPICS else None


def load_gold(path: str) -> tuple[pd.DataFrame, list[str], int]:
    """Returns (valid rows, problems, pending count)."""
    if str(path).lower().endswith((".xlsx", ".xls")):
        raw = pd.read_excel(path, sheet_name="Labels", dtype=str).fillna("")
    else:
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    need = {"item", "relevant", "topic", "sentiment", "epu", "ambiguous", "note"}
    if missing := need - set(raw.columns):
        raise ValueError(f"labels file is missing columns: {sorted(missing)}")
    rows, problems, pending = [], [], 0
    for r in raw.itertuples():
        if not any(_norm(x) for x in (r.relevant, r.topic, r.sentiment, r.epu)):
            pending += 1
            continue
        rel, epu, amb = _bool(r.relevant), _bool(r.epu), _bool(r.ambiguous) or False
        if rel is None:
            problems.append(f"{r.item}: 'relevant' must be y/n, got {r.relevant!r}")
            continue
        if rel:
            topic, sent = _topic(r.topic), _SENT.get(_norm(r.sentiment))
            if topic is None:
                problems.append(f"{r.item}: relevant but topic {r.topic!r} is not one of {TOPICS}")
                continue
            if sent is None:
                problems.append(f"{r.item}: relevant but sentiment {r.sentiment!r} invalid")
                continue
            if epu is None:
                problems.append(f"{r.item}: relevant but 'epu' must be y/n, got {r.epu!r}")
                continue
        else:
            if epu:
                problems.append(f"{r.item}: not relevant but epu=y (irrelevant is never EPU)")
                continue
            topic, sent, epu = "none", None, False
        rows.append({"item": r.item, "g_relevant": rel, "g_topic": topic,
                     "g_sentiment": sent, "g_epu": bool(epu), "ambiguous": amb, "note": r.note})
    return pd.DataFrame(rows), problems, pending

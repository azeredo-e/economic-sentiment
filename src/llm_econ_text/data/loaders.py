"""Loaders for the original Portuguese full-text files (2_textos_completos).

The raw files have no headline column: the headline is recovered from the article URL slug
(no accents) and the lead from the first paragraphs of the body, after the date/byline lines.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd

RAW_DIR = Path("data/raw/original/2_textos_completos")
TOPIC_GLOBS = {
    "agronegocio": "*agro*",
    "industria": "*industria*",
    "mercado_financeiro": "*mercado_financeiro*",
    "mercado_trabalho": "*mercado_trabalho*",
    "servicos": "*servi*",
}

_DATE_RE = re.compile(r"^\d{2}/\d{2}/\d{4}(\s+\d{1,2}h\d{2})?$")
_META_PREFIXES = ("- atualizado", "atualizado em", "do g1", "por ", "com informações", "colaborou")


def headline_from_url(url: str) -> str:
    """Best-effort headline from a G1 URL (search-click URLs carry the article in `u=`)."""
    target = parse_qs(urlparse(url).query).get("u", [url])[0]
    slug = unquote(urlparse(target).path).rstrip("/").rsplit("/", 1)[-1]
    slug = re.sub(r"\.(html|ghtml)$", "", slug)
    words = slug.replace("-", " ").strip()
    return words[:1].upper() + words[1:]


def _is_meta(line: str) -> bool:
    low = line.lower()
    return bool(_DATE_RE.match(line)) or (len(line) < 90 and low.startswith(_META_PREFIXES))


def extract_lead(text: str, max_chars: int = 600) -> str:
    """First paragraphs of the body, skipping leading date/byline lines."""
    lines = [ln.strip() for ln in str(text).split("\n") if ln.strip()]
    i = 0
    while i < len(lines) and _is_meta(lines[i]):
        i += 1
    out: list[str] = []
    size = 0
    for ln in lines[i:]:
        if out and size + len(ln) > max_chars:
            break
        out.append(ln)
        size += len(ln)
        if size >= max_chars:
            break
    return " ".join(out)[:max_chars]


def load_topic(topic: str, raw_dir: Path = RAW_DIR, max_chars: int = 600) -> pd.DataFrame:
    """Columns: article_id, source_topic, date, url, headline, lead."""
    matches = sorted(raw_dir.glob(TOPIC_GLOBS[topic] + ".xlsx"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one file for {topic} in {raw_dir}, got {matches}")
    df = pd.read_excel(matches[0])
    df = df.dropna(subset=["Texto"]).reset_index().rename(columns={"index": "row"})
    return pd.DataFrame(
        {
            "article_id": [f"{topic}-{r}" for r in df["row"]],
            "source_topic": topic,
            "date": pd.to_datetime(df["Data"]),
            "url": df["endereco"],
            "headline": [headline_from_url(u) for u in df["endereco"]],
            "lead": [extract_lead(t, max_chars) for t in df["Texto"]],
        }
    )

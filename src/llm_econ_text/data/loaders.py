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
_UPDATED_RE = re.compile(r"^(?:-\s*)?atualizado(?:\s+em)?$", re.IGNORECASE)
_LEGACY_META_PREFIXES = ("- atualizado", "atualizado em", "do g1", "por ", "com informações",
                         "colaborou", "edição do dia", "fique por")
_CAPTION_RE = re.compile(r"(?:[—–-]\s*Foto:|^Foto:|Crédito:|^Reprodução|\(Foto:)", re.IGNORECASE)
_PROMO_RE = re.compile(r"^(?:veja|leia|vídeos?|assista|saiba mais)\b", re.IGNORECASE)
# a reporter / agency / place line left after the "Atualizado" stanza, e.g. "Brasília, DF"
_NAME_LINE_RE = re.compile(r"^[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\w.,'’-]*(?:\s+[\w.,'’-]+){0,6}$")
_BYLINE_RE = re.compile(
    r"^(?:do|da|de|por|com informações)\s+(?:g1|globo|tv|rede|agência|agencia|redação|governo|"
    r"assessoria|reuters|afp|efe|bbc|folhapress|estadão)\b", re.IGNORECASE)
_SENTENCE_END = (".", "!", "?", ":", ";", "”", '"')


def headline_from_url(url: str) -> str:
    """Best-effort headline from a G1 URL (search-click URLs carry the article in `u=`)."""
    target = parse_qs(urlparse(url).query).get("u", [url])[0]
    slug = unquote(urlparse(target).path).rstrip("/").rsplit("/", 1)[-1]
    slug = re.sub(r"\.(html|ghtml)$", "", slug)
    words = slug.replace("-", " ").strip()
    return words[:1].upper() + words[1:]


def _is_byline_like(ln: str) -> bool:
    """Reporter / agency / place / sponsor line that follows the 'Atualizado' stanza."""
    if ln.endswith(_SENTENCE_END):
        return False
    if len(ln) < 90 and (_BYLINE_RE.match(ln)
                         or ln.lower().endswith("conteúdo de responsabilidade do anunciante")):
        return True
    return 6 <= len(ln) < 45 and bool(_NAME_LINE_RE.match(ln))


def _head_length(lines: list[str]) -> int:
    """Number of leading metadata lines to drop.

    Since 2010 G1 pages put an "Atualizado" stanza (bare word, "- Atualizado em", plus dates)
    after a byline block; from 2017 the block also starts with "fique por / dentro" and may end
    with reporter and place lines. Anchor on that stanza; fall back to prefix matching.
    """
    k = next((i for i, ln in enumerate(lines[:12]) if _UPDATED_RE.match(ln)), None)
    if k is not None:
        i = k + 1
        while i < len(lines) and _DATE_RE.match(lines[i]):
            i += 1
        skipped = 0
        while i < len(lines) - 1 and skipped < 3 and _is_byline_like(lines[i]):
            i += 1
            skipped += 1
        return i
    i = 0
    while i < len(lines) and (
        _DATE_RE.match(lines[i])
        or (len(lines[i]) < 90 and lines[i].lower().startswith(_LEGACY_META_PREFIXES))
    ):
        i += 1
    return i


def extract_lead(text: str, max_chars: int = 600) -> str:
    """First ~max_chars of the article body: metadata, photo captions and promo lines removed,
    hyperlink-split fragments rejoined, cut at a sentence end (never mid-word)."""
    lines = [ln.strip() for ln in str(text).split("\n") if ln.strip()]
    lines = lines[_head_length(lines):]
    body = [ln for ln in lines[:60]
            if not _CAPTION_RE.search(ln) and not (len(ln) < 100 and _PROMO_RE.match(ln))]
    out = " ".join(body)
    out = re.sub(r"\s+([,.;:!?)\]])", r"\1", out)
    out = re.sub(r"([(\[])\s+", r"\1", out)
    out = re.sub(r"\s+", " ", out).strip()
    if len(out) <= max_chars:
        return out
    cut = out[:max_chars]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    if end >= int(max_chars * 0.4):
        return cut[: end + 1]
    return cut[: cut.rfind(" ")].rstrip(",;:") + "…" if " " in cut else cut


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

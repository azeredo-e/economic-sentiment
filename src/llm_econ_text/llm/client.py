"""Cost-aware, cached LLM scorer.

Every call is logged to SQLite (tokens, cost, latency, status). Successful labels are cached by
(article_id, model, prompt_hash), so reruns are free. A budget guard aborts before the total
recorded spend would pass `total_usd * hard_stop_fraction`.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import anthropic
from pydantic import ValidationError

from llm_econ_text.llm.schema import ArticleLabel

# USD per 1M tokens: (input, output, cache read). Source: Anthropic price table, cached 2026-11-03.
# Haiku's cache-read rate is assumed at 10% of input (conservative); the Console is the truth.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10),
    "claude-opus-5-5": (4.00, 20.00, 0.20),
}
CACHE_WRITE_MULT = 1.25  # 5-minute cache writes


class BudgetExceeded(RuntimeError):
    pass


def compute_cost(model: str, input_tokens: int, cache_write: int, cache_read: int, output: int):
    p_in, p_out, p_cr = PRICES[model]
    return (
        input_tokens * p_in + cache_write * p_in * CACHE_WRITE_MULT + cache_read * p_cr
        + output * p_out
    ) / 1_000_000


def article_text(headline: str, lead: str) -> str:
    return f"Título (do URL): {headline}\n\nLide:\n{lead}"


@dataclass
class ScoreResult:
    article_id: str
    label: ArticleLabel | None
    cached: bool
    cost_usd: float
    status: str  # ok | invalid | cached


class Scorer:
    def __init__(
        self,
        system_prompt: str,
        prompt_version: str,
        model: str = "claude-haiku-5-5",
        effort: str = "low",
        db_path: str | Path = "data/processed/llm_cache.sqlite",
        total_usd: float = 10.0,
        hard_stop_fraction: float = 0.9,
        client: anthropic.Anthropic | None = None,
        max_tokens: int = 2048,
    ):
        self.system_prompt = system_prompt
        self.prompt_version = prompt_version
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.limit = total_usd * hard_stop_fraction
        self._client = client
        schema = json.dumps(anthropic.transform_schema(ArticleLabel), sort_keys=True)
        self.schema = json.loads(schema)
        self.prompt_hash = hashlib.sha256(
            f"{system_prompt}|{effort}|{schema}".encode()
        ).hexdigest()[:16]
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path)
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS calls (
                id INTEGER PRIMARY KEY, article_id TEXT, model TEXT, prompt_version TEXT,
                prompt_hash TEXT, ts REAL, input_tokens INT, cache_write_tokens INT,
                cache_read_tokens INT, output_tokens INT, cost_usd REAL, latency_s REAL,
                status TEXT, label_json TEXT, error TEXT)"""
        )
        self.db.commit()

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        return self._client

    def total_spent(self) -> float:
        return self.db.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM calls").fetchone()[0]

    def _cached(self, article_id: str) -> ArticleLabel | None:
        row = self.db.execute(
            "SELECT label_json FROM calls WHERE article_id=? AND model=? AND prompt_hash=? "
            "AND status='ok' LIMIT 1",
            (article_id, self.model, self.prompt_hash),
        ).fetchone()
        return ArticleLabel.model_validate_json(row[0]) if row else None

    def _log(self, article_id, usage, cost, latency, status, label_json=None, error=None):
        self.db.execute(
            "INSERT INTO calls (article_id, model, prompt_version, prompt_hash, ts, "
            "input_tokens, cache_write_tokens, cache_read_tokens, output_tokens, cost_usd, "
            "latency_s, status, label_json, error) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (article_id, self.model, self.prompt_version, self.prompt_hash, time.time(),
             *usage, cost, latency, status, label_json, error),
        )
        self.db.commit()

    def score(self, article_id: str, headline: str, lead: str) -> ScoreResult:
        hit = self._cached(article_id)
        if hit is not None:
            return ScoreResult(article_id, hit, True, 0.0, "cached")
        if self.total_spent() >= self.limit:
            raise BudgetExceeded(
                f"recorded spend ${self.total_spent():.4f} reached hard stop ${self.limit:.2f}"
            )
        t0 = time.time()
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[{"type": "text", "text": self.system_prompt,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": article_text(headline, lead)}],
            output_config={"effort": self.effort,
                           "format": {"type": "json_schema", "schema": self.schema}},
        )
        latency = time.time() - t0
        u = resp.usage
        usage = (u.input_tokens, u.cache_creation_input_tokens or 0,
                 u.cache_read_input_tokens or 0, u.output_tokens)
        cost = compute_cost(self.model, *usage)
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            label = ArticleLabel.model_validate_json(text)
        except ValidationError as e:
            self._log(article_id, usage, cost, latency, "invalid", text, str(e)[:500])
            return ScoreResult(article_id, None, False, cost, "invalid")
        self._log(article_id, usage, cost, latency, "ok", label.model_dump_json())
        return ScoreResult(article_id, label, False, cost, "ok")

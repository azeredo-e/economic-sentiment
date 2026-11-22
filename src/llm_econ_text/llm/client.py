"""Cost-aware, cached LLM scorer.

Every call is logged to SQLite (tokens, cost, latency, status). Successful labels are cached by
(article_id, model, prompt_hash), so reruns are free. A budget guard aborts before the total
recorded spend would pass `total_usd * hard_stop_fraction`. Bulk work goes through the Batch API
(50% price); `submit_batch` / `collect_batch` are split so a crash never loses a paid batch.
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

from llm_econ_text.llm.schema import ArticleLabel, normalize_label

# USD per 1M tokens: (input, output, cache read). Source: Anthropic price table, cached 2026-11-03.
# Haiku's cache-read rate is assumed at 10% of input (conservative); the Console is the truth.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10),
    "claude-opus-5-5": (4.00, 20.00, 0.20),
}
CACHE_WRITE_MULT = 1.25  # 5-minute cache writes
BATCH_DISCOUNT = 0.5
EST_BATCH_COST_PER_ARTICLE = 0.00012  # measured ~0.00014 at full price; batch is half
SAFETY = 2.0  # budget check multiplies the estimate by this


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


def input_hash(headline: str, lead: str) -> str:
    """The cache must notice when the INPUT text changes (e.g. after a loader fix)."""
    return hashlib.sha256(article_text(headline, lead).encode()).hexdigest()[:16]


@dataclass
class ScoreResult:
    article_id: str
    label: ArticleLabel | None
    cached: bool
    cost_usd: float
    status: str  # ok | invalid | cached
    coerced: bool = False  # irrelevant article whose sentiment/EPU were reset (see schema)


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
        self.db = sqlite3.connect(db_path, timeout=60)
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS calls (
                id INTEGER PRIMARY KEY, article_id TEXT, model TEXT, prompt_version TEXT,
                prompt_hash TEXT, ts REAL, input_tokens INT, cache_write_tokens INT,
                cache_read_tokens INT, output_tokens INT, cost_usd REAL, latency_s REAL,
                status TEXT, label_json TEXT, error TEXT)"""
        )
        for col in ("batch_id TEXT", "input_hash TEXT"):
            try:
                self.db.execute(f"ALTER TABLE calls ADD COLUMN {col}")
            except sqlite3.OperationalError:
                pass  # column already exists
        self.db.execute("CREATE TABLE IF NOT EXISTS batch_items "
                        "(batch_id TEXT, article_id TEXT, input_hash TEXT)")
        self.db.commit()

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        return self._client

    def total_spent(self) -> float:
        return self.db.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM calls").fetchone()[0]

    def _cached(self, article_id: str, ihash: str) -> ArticleLabel | None:
        row = self.db.execute(
            "SELECT label_json FROM calls WHERE article_id=? AND model=? AND prompt_hash=? "
            "AND input_hash=? AND status='ok' LIMIT 1",
            (article_id, self.model, self.prompt_hash, ihash),
        ).fetchone()
        return ArticleLabel.model_validate_json(row[0]) if row else None

    def get_label(self, article_id: str, headline: str, lead: str):
        """Cached label (normalized) and whether it was coerced; None if not scored for
        exactly this input."""
        hit = self._cached(article_id, input_hash(headline, lead))
        return normalize_label(hit) if hit is not None else None

    def _params(self, headline: str, lead: str) -> dict:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [{"type": "text", "text": self.system_prompt,
                        "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": article_text(headline, lead)}],
            "output_config": {"effort": self.effort,
                              "format": {"type": "json_schema", "schema": self.schema}},
        }

    def _log(self, article_id, usage, cost, latency, status, label_json=None, error=None,
             batch_id=None, ihash=None):
        self.db.execute(
            "INSERT INTO calls (article_id, model, prompt_version, prompt_hash, ts, "
            "input_tokens, cache_write_tokens, cache_read_tokens, output_tokens, cost_usd, "
            "latency_s, status, label_json, error, batch_id, input_hash) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (article_id, self.model, self.prompt_version, self.prompt_hash, time.time(),
             *usage, cost, latency, status, label_json, error, batch_id, ihash),
        )
        self.db.commit()

    def _record(self, article_id, ihash, resp, latency, discount=1.0, batch_id=None) -> ScoreResult:
        u = resp.usage
        usage = (u.input_tokens, u.cache_creation_input_tokens or 0,
                 u.cache_read_input_tokens or 0, u.output_tokens)
        cost = compute_cost(self.model, *usage) * discount
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            label = ArticleLabel.model_validate_json(text)
        except ValidationError as e:
            self._log(article_id, usage, cost, latency, "invalid", text, str(e)[:500], batch_id,
                      ihash)
            return ScoreResult(article_id, None, False, cost, "invalid")
        self._log(article_id, usage, cost, latency, "ok", label.model_dump_json(),
                  batch_id=batch_id, ihash=ihash)  # raw label; normalized on read
        label, coerced = normalize_label(label)
        return ScoreResult(article_id, label, False, cost, "ok", coerced)

    def _check_budget(self, planned: float = 0.0) -> None:
        if self.total_spent() + planned >= self.limit:
            raise BudgetExceeded(
                f"recorded spend ${self.total_spent():.4f} + planned ${planned:.4f} "
                f"would reach hard stop ${self.limit:.2f}"
            )

    def score(self, article_id: str, headline: str, lead: str) -> ScoreResult:
        ihash = input_hash(headline, lead)
        hit = self._cached(article_id, ihash)
        if hit is not None:
            label, coerced = normalize_label(hit)
            return ScoreResult(article_id, label, True, 0.0, "cached", coerced)
        self._check_budget()
        t0 = time.time()
        resp = self.client.messages.create(**self._params(headline, lead))
        return self._record(article_id, ihash, resp, time.time() - t0)

    # ---- Batch API -------------------------------------------------------------------------
    def submit_batch(self, items: list[tuple[str, str, str]]) -> str | None:
        """items: (article_id, headline, lead). Skips ones already scored for this exact input."""
        todo = [(aid, h, ld, input_hash(h, ld)) for aid, h, ld in items
                if self._cached(aid, input_hash(h, ld)) is None]
        if not todo:
            return None
        self._check_budget(len(todo) * EST_BATCH_COST_PER_ARTICLE * SAFETY)
        batch = self.client.messages.batches.create(
            requests=[{"custom_id": aid, "params": self._params(h, ld)} for aid, h, ld, _ in todo]
        )
        self.db.executemany("INSERT INTO batch_items VALUES (?,?,?)",
                            [(batch.id, aid, ih) for aid, _, _, ih in todo])
        self.db.commit()
        return batch.id

    def wait_batch(self, batch_id: str, poll_s: float = 30.0):
        while True:
            b = self.client.messages.batches.retrieve(batch_id)
            if b.processing_status == "ended":
                return b
            time.sleep(poll_s)

    def collect_batch(self, batch_id: str) -> dict[str, int]:
        """Log every result of an ended batch, once. Returns {status: count}."""
        hashes = dict(self.db.execute(
            "SELECT article_id, input_hash FROM batch_items WHERE batch_id=?", (batch_id,)))
        done = {r[0] for r in self.db.execute(
            "SELECT article_id FROM calls WHERE batch_id=?", (batch_id,))}
        counts: dict[str, int] = {}
        for r in self.client.messages.batches.results(batch_id):
            aid, kind = r.custom_id, r.result.type
            if aid in done:  # re-collecting must not double-count (ok or invalid)
                status = "already_logged"
            elif kind == "succeeded":
                status = self._record(aid, hashes.get(aid), r.result.message, 0.0,
                                      BATCH_DISCOUNT, batch_id).status
            else:
                self._log(aid, (0, 0, 0, 0), 0.0, 0.0, f"batch_{kind}", None, None, batch_id,
                          hashes.get(aid))
                status = f"batch_{kind}"
            counts[status] = counts.get(status, 0) + 1
        return counts

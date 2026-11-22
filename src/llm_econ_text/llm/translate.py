"""Cached, cost-logged PT -> EN translation of headline + lead (M4 translation experiment).

Reuses the Scorer's ledger: each translation is one row with status 'translated', so the budget
guard sees its cost and reruns are free (cache key: article id + input text + prompt + model).
"""

from __future__ import annotations

import json
import time

import anthropic
from pydantic import BaseModel, ValidationError

from llm_econ_text.llm.client import Scorer, compute_cost, input_hash


class Translation(BaseModel):
    headline_en: str
    lead_en: str


class Translator:
    def __init__(self, system_prompt: str, version: str = "translate_v1", **scorer_kwargs):
        # A Scorer instance is used only for its ledger, client and budget guard.
        self.s = Scorer(system_prompt, version, **scorer_kwargs)
        self.schema = anthropic.transform_schema(Translation)

    def _params(self, headline: str, lead: str) -> dict:
        return {
            "model": self.s.model,
            "max_tokens": 4096,
            "system": [{"type": "text", "text": self.s.system_prompt,
                        "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": json.dumps(
                {"headline": headline, "lead": lead}, ensure_ascii=False)}],
            "output_config": {"effort": self.s.effort,
                              "format": {"type": "json_schema", "schema": self.schema}},
        }

    def translate(self, article_id: str, headline: str, lead: str) -> Translation | None:
        ih = input_hash(headline, lead)
        row = self.s.db.execute(
            "SELECT label_json FROM calls WHERE article_id=? AND model=? AND prompt_hash=? "
            "AND input_hash=? AND status='translated' LIMIT 1",
            (article_id, self.s.model, self.s.prompt_hash, ih)).fetchone()
        if row:
            return Translation.model_validate_json(row[0])
        self.s._check_budget()
        t0 = time.time()
        resp = self.s.client.messages.create(**self._params(headline, lead))
        latency = time.time() - t0
        u = resp.usage
        usage = (u.input_tokens, u.cache_creation_input_tokens or 0,
                 u.cache_read_input_tokens or 0, u.output_tokens)
        cost = compute_cost(self.s.model, *usage)
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            tr = Translation.model_validate_json(text)
        except ValidationError as e:
            self.s._log(article_id, usage, cost, latency, "translation_invalid", text,
                        str(e)[:300], None, ih)
            return None
        self.s._log(article_id, usage, cost, latency, "translated", tr.model_dump_json(), None,
                    None, ih)
        return tr

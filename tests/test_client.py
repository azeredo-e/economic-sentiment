from types import SimpleNamespace

import pytest

from llm_econ_text.llm.client import BudgetExceeded, Scorer, compute_cost

GOOD = (
    '{"relevant": true, "topic": "industria", "sentiment": -0.5, "sentiment_label": "negative",'
    ' "epu": false, "epu_reason": null, "confidence": 0.9, "rationale": "Queda da produção."}'
)
BAD = GOOD.replace('"negative"', '"positive"')  # label inconsistent with score


class FakeClient:
    def __init__(self, texts, out_tokens=100):
        self.texts = list(texts)
        self.calls = 0
        self.out_tokens = out_tokens
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls += 1
        usage = SimpleNamespace(
            input_tokens=1000, cache_creation_input_tokens=0,
            cache_read_input_tokens=0, output_tokens=self.out_tokens,
        )
        block = SimpleNamespace(type="text", text=self.texts.pop(0))
        return SimpleNamespace(usage=usage, content=[block])


def make(tmp_path, client, **kw):
    return Scorer("sys", "v1", db_path=tmp_path / "c.sqlite", client=client, **kw)


def test_compute_cost_haiku():
    assert compute_cost("claude-haiku-5-5", 1_000_000, 0, 0, 1_000_000) == pytest.approx(0.60)


def test_cache_makes_rerun_free(tmp_path):
    fake = FakeClient([GOOD])
    s = make(tmp_path, fake)
    first = s.score("a1", "h", "l")
    second = s.score("a1", "h", "l")
    assert first.status == "ok" and second.status == "cached"
    assert fake.calls == 1
    assert first.cost_usd > 0
    assert s.total_spent() == pytest.approx(first.cost_usd)


def test_invalid_label_is_flagged_and_billed(tmp_path):
    s = make(tmp_path, FakeClient([BAD]))
    r = s.score("a1", "h", "l")
    assert r.status == "invalid" and r.label is None
    assert s.total_spent() > 0


def test_budget_hard_stop(tmp_path):
    fake = FakeClient([GOOD, GOOD], out_tokens=1_000_000)  # one call costs ~$0.50
    s = make(tmp_path, fake, total_usd=0.5, hard_stop_fraction=0.9)
    s.score("a1", "h", "l")
    with pytest.raises(BudgetExceeded):
        s.score("a2", "h", "l")
    assert fake.calls == 1


def test_prompt_change_invalidates_cache(tmp_path):
    f1, f2 = FakeClient([GOOD]), FakeClient([GOOD])
    make(tmp_path, f1).score("a1", "h", "l")
    other = Scorer("different", "v2", db_path=tmp_path / "c.sqlite", client=f2)
    assert other.score("a1", "h", "l").status == "ok" and f2.calls == 1

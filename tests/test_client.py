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


IRRELEVANT_EPU = (
    '{"relevant": false, "topic": "none", "sentiment": 0.0, "sentiment_label": "neutral",'
    ' "epu": true, "epu_reason": "Renúncia no Peru.", "confidence": 0.6, "rationale": "Exterior."}'
)


def test_scorer_normalizes_but_ledger_keeps_raw(tmp_path):
    s = make(tmp_path, FakeClient([IRRELEVANT_EPU]))
    r = s.score("a1", "h", "l")
    assert r.coerced and r.label.epu is False
    raw = s.db.execute("SELECT label_json FROM calls").fetchone()[0]
    assert '"epu":true' in raw.replace(" ", "")
    again = s.score("a1", "h", "l")  # cache read is normalized too
    assert again.status == "cached" and again.coerced and again.label.epu is False


class FakeBatches:
    def __init__(self, texts):
        self.texts, self.created = texts, []
        self.requests = []

    def create(self, requests):
        self.requests = requests
        return SimpleNamespace(id="batch_1")

    def retrieve(self, batch_id):
        return SimpleNamespace(processing_status="ended", request_counts=None)

    def results(self, batch_id):
        for req, text in zip(self.requests, self.texts):
            usage = SimpleNamespace(input_tokens=1000, cache_creation_input_tokens=0,
                                    cache_read_input_tokens=0, output_tokens=100)
            msg = SimpleNamespace(usage=usage, content=[SimpleNamespace(type="text", text=text)])
            yield SimpleNamespace(custom_id=req["custom_id"],
                                  result=SimpleNamespace(type="succeeded", message=msg))


def test_batch_roundtrip_half_price_and_idempotent(tmp_path):
    fake = SimpleNamespace(messages=SimpleNamespace(batches=FakeBatches([GOOD, GOOD])))
    s = make(tmp_path, fake)
    full = compute_cost("claude-haiku-5-5", 1000, 0, 0, 100)
    assert s.submit_batch([("a1", "h", "l"), ("a2", "h", "l")]) == "batch_1"
    assert s.collect_batch("batch_1") == {"ok": 2}
    assert s.total_spent() == pytest.approx(2 * full * 0.5)
    assert s.collect_batch("batch_1") == {"already_logged": 2}  # no double billing
    assert s.submit_batch([("a1", "h", "l")]) is None  # cached, nothing to submit
    assert s.get_label("a1", "h", "l")[0].topic == "industria"


def test_batch_budget_check_blocks_oversized_submit(tmp_path):
    fake = SimpleNamespace(messages=SimpleNamespace(batches=FakeBatches([])))
    s = make(tmp_path, fake, total_usd=0.001, hard_stop_fraction=0.9)
    with pytest.raises(BudgetExceeded):
        s.submit_batch([(f"a{i}", "h", "l") for i in range(100)])
    assert fake.messages.batches.requests == []


def test_cache_key_includes_the_input_text(tmp_path):
    fake = FakeClient([GOOD, GOOD])
    s = make(tmp_path, fake)
    assert s.score("a1", "h", "old lead").status == "ok"
    assert s.score("a1", "h", "old lead").status == "cached"
    assert s.score("a1", "h", "NEW cleaned lead").status == "ok"  # input changed -> new call
    assert fake.calls == 2
    assert s.get_label("a1", "h", "NEW cleaned lead") is not None
    assert s.get_label("a1", "h", "something else") is None


def test_recollecting_a_batch_never_double_counts_even_invalid(tmp_path):
    fake = SimpleNamespace(messages=SimpleNamespace(batches=FakeBatches([BAD, GOOD])))
    s = make(tmp_path, fake)
    s.submit_batch([("x1", "h", "l"), ("x2", "h", "l")])
    first = s.collect_batch("batch_1")
    spent = s.total_spent()
    second = s.collect_batch("batch_1")
    assert first == {"invalid": 1, "ok": 1}
    assert second == {"already_logged": 2}
    assert s.total_spent() == pytest.approx(spent)

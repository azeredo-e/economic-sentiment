import pytest
from pydantic import ValidationError

from llm_econ_text.llm.schema import ArticleLabel


def make(**kw):
    base = {
        "relevant": True,
        "topic": "industria",
        "sentiment": -0.6,
        "sentiment_label": "negative",
        "epu": False,
        "epu_reason": None,
        "confidence": 0.8,
        "rationale": "Queda da produção industrial.",
    }
    base.update(kw)
    return ArticleLabel(**base)


def test_valid():
    assert make().topic == "industria"


def test_irrelevant_requires_none_topic():
    with pytest.raises(ValidationError):
        make(relevant=False)
    assert make(relevant=False, topic="none").relevant is False


def test_label_must_match_score():
    with pytest.raises(ValidationError):
        make(sentiment=0.9)
    assert make(sentiment=0.1, sentiment_label="neutral").sentiment_label == "neutral"


def test_epu_needs_reason():
    with pytest.raises(ValidationError):
        make(epu=True)
    assert make(epu=True, epu_reason="Incerteza sobre o teto de gastos.").epu


def test_score_bounds():
    with pytest.raises(ValidationError):
        make(sentiment=1.5)


def test_normalize_resets_irrelevant_epu_and_sentiment():
    from llm_econ_text.llm.schema import normalize_label

    raw = make(relevant=False, topic="none", sentiment=0.4, sentiment_label="positive",
               epu=True, epu_reason="Incerteza política no Peru.")
    fixed, coerced = normalize_label(raw)
    assert coerced
    assert (fixed.sentiment, fixed.sentiment_label, fixed.epu, fixed.epu_reason) == (
        0.0, "neutral", False, None)


def test_normalize_leaves_relevant_and_clean_irrelevant_alone():
    from llm_econ_text.llm.schema import normalize_label

    assert normalize_label(make(epu=True, epu_reason="x"))[1] is False
    clean = make(relevant=False, topic="none", sentiment=0.0, sentiment_label="neutral")
    assert normalize_label(clean) == (clean, False)

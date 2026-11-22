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

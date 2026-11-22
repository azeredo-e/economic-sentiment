import pytest

from llm_econ_text.eval import dictionary as d

LEX = {"good": 1, "growth": 1, "bad": -1, "crisis": -1}


def test_polarity_and_normalisation():
    assert d.score("good growth", shifter=False, lex=LEX) == 1.0
    assert d.score("bad good growth", shifter=False, lex=LEX) == pytest.approx(1 / 3)
    assert d.score("", shifter=False, lex=LEX) == 0.0


def test_negator_flips_only_with_the_shifter():
    assert d.score("not good", shifter=True, lex=LEX) < 0
    assert d.score("not good", shifter=False, lex=LEX) > 0


def test_amplifier_and_deamplifier():
    assert d.score("very good", shifter=True, lex=LEX) > d.score("good", shifter=True, lex=LEX) / 2 * 1.5
    assert d.score("slightly bad", shifter=True, lex=LEX) > d.score("bad", shifter=True, lex=LEX) / 2 * 1.5 - 1


def test_stopword_stripping_erases_the_negator_and_the_flip():
    assert "not" in d.stopwords() and "no" in d.stopwords() and "against" in d.stopwords()
    kept = d.score("this is not good", shifter=True, strip_stopwords=False, lex=LEX)
    stripped = d.score("this is not good", shifter=True, strip_stopwords=True, lex=LEX)
    assert kept < 0 < stripped  # "not good" scored as "good" once the stop word is gone

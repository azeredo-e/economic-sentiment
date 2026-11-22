# M4: does machine translation change what the LLM measures?

Status: first run, 2026-11-22. Numbers are from `results/translation_effect.json`; reproduce with
`translate_gold`, `predict_gold` (three runs) and `translation_effect` (see the end).

## Setup
- 300 gold articles (headline + lead), accepted labels = Claude-labelled, owner-accepted (provisional).
- Same model (Haiku 5.5), same prompt (v2), three runs:
  **PT** (Portuguese, as in the pipeline), **PT2** (the same Portuguese scored a second time, the
  noise control) and **EN** (the same articles translated to English by Haiku with a
  faithful-not-improved prompt, `prompts/translate_v1.md`, then scored with the *unchanged* v2 prompt).
- Temperature cannot be set on these models, so repeated scoring of identical text differs. The
  translation effect is therefore flip(PT, EN) **minus** flip(PT, PT2), never flip(PT, EN) alone.
- Flip rates are unweighted counts on the 300-article sample (oversampled strata), not corpus rates.

## Results
| Label | Noise: PT vs PT2 | Translation: PT vs EN | Difference (95% CI) |
|---|---|---|---|
| relevance flips | 6.0% | 10.3% | **+4.3 pts (1.0 to 7.7)** |
| topic flips (both relevant) | 4.2% | 6.5% | +2.3 (-0.3 to 5.0) |
| sentiment-label flips (both relevant) | 7.1% | 9.0% | +1.9 (-3.0 to 7.0) |
| EPU flips | 4.0% | 4.7% | +0.7 (-1.0 to 2.3) |

Sentiment *score* (continuous), both relevant: mean absolute shift 0.062 (noise 0.051), Spearman
0.963 (noise 0.974), mean signed shift +0.012 (no systematic bias), share with |shift| >= 0.2:
10.5% (noise 5.2%).

Agreement with the accepted labels (weighted):
| | PT | PT2 | EN |
|---|---|---|---|
| relevance macro-F1 | 0.914 | 0.919 | 0.874 |
| sentiment macro-F1 | 0.750 | 0.775 | 0.740 |
| EPU F1 (31 positives) | 0.777 | 0.612 | 0.720 |

Reading: only **relevance** shows a translation effect distinguishable from noise, and in one
direction (English is more conservative: 19 relevant-to-irrelevant flips against 12 the other way,
versus 7 and 11 between the two Portuguese runs). Topic, sentiment and EPU changes are within noise.

**Noise is a finding in itself.** Scoring identical text twice flips 4-7% of labels, and EPU F1
moved by 0.17 between two identical runs. Differences smaller than about 0.03 (sentiment) or
0.15 (EPU F1) between any two prompts or models on this gold set cannot be trusted.

## Error taxonomy (38 "robust" flips: PT and PT2 agree, EN differs)
Classified by hand from the Portuguese and English text side by side; one rater, so treat the
categories as a reading, not a measurement. 84% of these items were ones the labeller had already
flagged as unsure (58% overall), i.e. they sit on the guide's decision boundary.

| Kind | n | What it is |
|---|---|---|
| Boundary relevance flip | 16 | Faithful translation, borderline article (events, training programmes, political speeches, culture with an economic hook). 11 relevant-to-irrelevant, 5 the reverse. No meaning lost; wording changes tip an unstable call |
| Sentiment threshold crossing | 12 | Score moves by <= 0.3 across the +/-0.2 neutral band (16 of the 18 sentiment-label flips overall; only 3 true sign reversals of >= 0.2). An artefact of discretising a stable score |
| Topic flip | 4 | All items whose topic fit was already weak (trade balance, merger, financing, employment index) |
| EPU flip | 5 | 3 lost, 2 gained; low base rate, same size as noise |
| Translation failure | 2 | One lead returned as "x", one abbreviation ("ES") left untranslated in the headline. 0.7% of all translations |
| Idiom rendered literally | 1 | "Estamos indo pro vinagre" became "heading for the vinegar", losing the negative meaning; the item flipped from relevant and negative to not relevant |

(Two items change two labels and appear in two rows, so the rows add to 40 for 38 items.)

**Not observed:** no flip traced to lost negation, lost hedging or number errors. That is expected
here, because the translation prompt explicitly preserves them (a best case for translation).

## Finding about the original pipeline (not caused by translation itself)
The thesis' `Tratamento.py` removes NLTK stop words in Portuguese and in English *before* scoring.
NLTK's lists contain the negators (`não`, `nem`, `sem`; `not`, `no`, `nor`, `against`). The thesis'
sentiment step then relies on sentometrics' English valence shifters (negators among them).
Measured on the 16,026 cleaned Mercado Financeiro articles, matched to their raw text by URL:

| Negators per 1,000 words | raw PT | cleaned PT | cleaned EN | (reference) faithful EN, gold leads |
|---|---|---|---|---|
| value | 5.9 | 1.2 | 2.3 | 3.9 (PT leads: 4.0) |

Concrete case in the data: "nenhum sacrifício" appears as "sacrifice" in the English column.
The rates are per *remaining* word, so the true loss is larger than shown. The effect on
agreement with the gold labels was then tested (`docs/dictionary_baseline.md`): negators sit
directly before a sentiment word in only 3% of headline-plus-lead texts, and stripping them changed
no measurable agreement. The mechanism is real, but not shown to matter at that length.

## Limitations
- The LLM translation is higher fidelity than Google Translate, which the thesis used: these
  results are a lower bound on machine-translation damage, and the Google arm is not run.
- The v2 prompt says the article is in Portuguese while the English arm is English. An
  English-adapted prompt is an untested control that could explain part of the relevance shift.
- Labels are provisional and the effective sample size is about 99, so intervals are wide.

## Reproduce
```bash
uv run python -m llm_econ_text.eval.translate_gold
uv run python -m llm_econ_text.eval.predict_gold --name haiku_v2_pt_retest --id-suffix "#pt2"
uv run python -m llm_econ_text.eval.predict_gold --name haiku_v2_en --inputs data/gold/translations_haiku_v1.csv --id-suffix "#en"
uv run python -m llm_econ_text.eval.translation_effect
```

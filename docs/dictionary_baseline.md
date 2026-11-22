# Dictionary baseline: the thesis' method vs the accepted labels and vs the LLM

Status: first run, 2026-11-22 (`results/dictionary_baseline.json`, `results/preds_thesis_pipeline.csv`).
Labels are the Claude-labelled, owner-accepted gold set (provisional); effective n about 99.

## What was run
- **Original pipeline, all four tasks** (`preds_thesis_pipeline.csv`): relevance = survived the
  thesis' keyword filter and cleaning; topic = the search folder the article came from;
  sentiment = Loughran-McDonald (LM) dictionary plus a negation shifter on the thesis' **own cleaned
  English text** (full article, stop words stripped); EPU = its policy-and-uncertainty keyword rule.
  Covers the 200 of 300 gold items that survived its cleaning (items it dropped count as irrelevant).
- **Controlled sentiment arms** on our faithful English translation of the same headline + lead the
  LLM saw: `en_plain` (no shifter), `en_shifter`, and `en_stripped` (NLTK English stop words removed
  first, as the thesis did, then the shifter).
- Scored on the 211 gold-relevant items the LLM also called relevant, so both methods face the same
  articles. The neutral band is tuned on the dev split only and reported on test; Spearman is
  threshold-free.
- **Approximation:** R and sentometrics are not installed, so the shifter lists are my
  reconstruction (standard negators plus a few amplifiers), not the thesis' exact run.

## Sentiment (3-class), same 211 items
| Method | Spearman vs gold (95% CI) | test macro-F1 (n=139) | test kappa |
|---|---|---|---|
| **LLM (Haiku v2, Portuguese)** | **0.911 (0.887-0.928)** | **0.735** | **0.644** |
| Dictionary, plain | 0.491 (0.378-0.597) | 0.529 | 0.307 |
| Dictionary, shifter | 0.482 (0.364-0.591) | 0.540 | 0.341 |
| Dictionary, stop words stripped (thesis-like) | 0.490 (0.373-0.596) | 0.524 | 0.297 |
| Thesis' own cleaned text (n=147 items) | 0.464 (0.317-0.591) | 0.412 (n=98) | 0.167 |
| LLM on the same 147 items | 0.908 (0.875-0.928) | 0.699 (n=98) | 0.559 |

About 30% of headline-plus-lead texts contain no LM word at all (zero score).

## The original pipeline on all four tasks (weighted, effective n about 99)
| Task | Original pipeline | LLM (Haiku v2) |
|---|---|---|
| Relevance macro-F1 | 0.64 (0.53-0.71), kappa 0.28 | 0.91 (0.86-0.95), kappa 0.83 |
| Topic accuracy among gold-relevant | 0.38 (59% among the articles it kept) | 0.75 (93% when both call it relevant) |
| Sentiment macro-F1 | 0.44 (0.34-0.52) | 0.75 (0.67-0.83) |
| EPU F1 | 0.05 (precision 0.03, recall 0.46) | 0.78 (0.55-0.92) |

Read these as "different definitions" as much as "errors": the thesis' relevance is a keyword
filter and its topic is a retrieval folder, whereas the gold labels follow this project's guide.
The EPU result is the clearest: its rule flags 78 of 300 articles against 31 true positives
(see also the 37% corpus-wide hit rate).

## Hypothesis from the translation write-up: tested, not supported
`docs/translation_experiment.md` showed the thesis' stop-word list removes negators ("not", "no",
"nor"; "não", "nem", "sem"). Tested here on headline + lead:
- A negator directly before a sentiment word occurs in only **3.0%** of articles (9 of 300), so the
  shifter changes a score in 4.7%.
- Stripping stop words changes the 3-class label in 10% of articles (it also changes the word
  count and removes amplifiers), yet agreement with the gold labels does not move: Spearman 0.490
  stripped vs 0.482 with the shifter vs 0.491 without it, all within one interval of each other.
So the negator mechanism is real but **not shown to matter** at headline-plus-lead length. It could
matter more on full articles, but there are no full-article gold labels to test it against.

## What limits the dictionary instead
Coverage (30% of texts have no LM word) and definition: LM scores the tone of words in financial
filings, while the gold labels ask what an article implies for the Brazilian economy in its topic
("unemployment falls" is good news in negative-sounding words). That is a gap no shifter closes.

## Limitations
- Shifter lists are approximate; the thesis' exact sentometrics run is not reproduced.
- One gold set, provisional labels, wide intervals; the threshold is tuned on only 72 dev items.
- The original pipeline is scored against labels written for a different definition of relevance.

## Reproduce
```bash
uv run python -m llm_econ_text.eval.dictionary_baseline
uv run python -m llm_econ_text.eval.evaluate --name thesis_pipeline --pred results/preds_thesis_pipeline.csv
```
The LM dictionary file (`data/raw/original/lm/Loughran_McDonald_en.xlsx`) comes from the thesis repo
and keeps its own licence terms; it is gitignored with the other raw data.

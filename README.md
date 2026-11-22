# LLM-based textual indices for Brazilian macroeconomic forecasting

This project is an **extension** of Nathan Ramos de Almeida's undergraduate thesis,
*"O poder das palavras: usando texto para previsões econômicas"* (UFRGS, 2024; advisor
Prof. Dr. Hudson da Silva Torrent), and of its code at
<https://github.com/E30895/economics-forecast-using-text> (MIT License, © 2024 Nathan Almeida).
The news corpus, the economic dataset, the original TESI/TEPU indices and the original
forecasting results all come from that work and are **not** this project's own. What is new
here is the LLM-based replacement of the dictionary NLP, the evaluation against human labels,
the translation experiment, and the Python port of the forecasting step.

## What this project asks

1. Do LLM-scored indices agree better with human judgement than the original
   Loughran–McDonald dictionary (applied to machine-translated text)?
2. Do they improve the original forecasts of monthly Brazilian economic activity?
3. How much does machine translation (PT → EN) change measured sentiment?

## What changes relative to the thesis

| Step | Original | This project |
|---|---|---|
| Relevance filter | Keyword list per topic | LLM classification, structured output |
| Translation | Google Translate PT→EN | None: Portuguese scored directly (translation measured as an experiment) |
| Sentiment (TESI) | LM dictionary + shifter | LLM score from system prompt, few-shot examples, JSON schema |
| Uncertainty (TEPU) | "Policy" and "uncertainty" keywords | LLM judges genuine economic-policy uncertainty |
| Forecasting | LASSO / ENET / Boosting vs SARIMA, R | Same models, horizons and tests, ported to Python and validated against the original Table 3 |

## Design decisions

- **LLM provider:** Anthropic API.
- **Forecasting:** Python port. The port is only trusted after it reproduces the thesis'
  Table 3 with the *original* indices (milestone M1). Two known places where a Python port
  drifts from the thesis, both fixed by design (details in `docs/forecast_port_notes.md`):
  - **Boosting** is componentwise linear L2 boosting (thesis Eqs. 10–11), hand-written,
    **not** `sklearn.GradientBoostingRegressor` (tree boosting).
  - **SARIMA benchmark** must match R's `forecast::auto.arima` (BIC, Ljung-Box check).
    `pmdarima.auto_arima` searches a different model set, which would change every relative RMSE.
- **Corpus:** headline + lead paragraph, stratified sample (about 100 articles per topic per month).
- **Gold set:** about 300 hand-labelled articles (relevance, 3-class sentiment, EPU).

## Layout

```
configs/      model, prompt version, sample sizes
prompts/      versioned system prompts and few-shot examples
src/llm_econ_text/
  data/       Drive download, loaders for the original files
  llm/        client, Pydantic schema, retries, cost logging
  agent/      tools + orchestration (M5)
  eval/       metrics vs gold, translation experiment
  indices/    monthly aggregation to TESI / TEPU
  forecast/   Python port of the forecasting pipeline
data/         raw / gold / processed (raw is not committed)
docs/         labeling guide
```

## Setup

```bash
uv sync
cp .env.example .env   # add ANTHROPIC_API_KEY
uv run pytest
```

Original data (public Google Drive folder from the thesis repo):

```bash
uv run python -m llm_econ_text.data.download --list      # dry run, no download
uv run python -m llm_econ_text.data.download             # downloads into data/raw/original
```

## Data and redistribution

Raw scraped G1 articles are kept out of version control because redistribution rights are
unclear. This repository publishes code, prompts, gold labels (if permitted) and aggregated indices.

## References

- Almeida, N. R. (2024). *O poder das palavras: usando texto para previsões econômicas.* UFRGS.
- Aprigliano, V. et al. (2023). *International Journal of Forecasting*, 39(2), 791–808.
- Baker, S., Bloom, N., Davis, S. (2016). Measuring economic policy uncertainty. *QJE*, 131(4).
- Bantis, E., Clements, M., Urquhart, A. (2023). *International Journal of Forecasting*, 39(4).
- Loughran, T., McDonald, B. (2011). *Journal of Finance*, 66(1).
- Welch, I., Goyal, A. (2008). *Review of Financial Studies*, 21(4).

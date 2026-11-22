# Labeling guide (v0.2)

Status: decisions D1-D4 accepted on 2026-11-22 (the recommendations below are now the rules).
Written from the first 50 model-labelled articles; not yet validated against any gold labels.
Prompt `prompts/v2/system.md` implements this guide.

## 0. What you label, and what you see
- Article = **headline + lead**, exactly what the model sees. The headline is rebuilt from the
  URL slug (no accents, e.g. "Pais esta tranquilo quanto a febre aftosa diz ministro").
  The lead is the first ~600 characters of the body after the date/byline.
- Label **only from that text**. Do not open the full article, even if the lead is thin: the
  point is to measure the same input.
- **Label blind.** Do not look at any model output while labelling (it anchors you).
- The folder an article came from (the search keyword that retrieved it) is **not** its topic.

Each article gets five fields:
`relevant` (y/n) · `topic` · `sentiment` (negative / neutral / positive) · `epu` (y/n) ·
`ambiguous` (y/n) plus a free-text note.

## 1. Relevance and topic
**Relevant = the article's main subject is Brazilian economic activity within one of five
topics.** If it is relevant, pick the one topic it is mainly about; otherwise `none`.

| Topic | Includes |
|---|---|
| `industria` | manufacturing, mining and extraction, construction, utilities, industrial output/PMI, industrial policy (e.g. a tax cut for car makers) |
| `agronegocio` | crops, livestock, harvests, agricultural commodity prices, agro exports, rural credit |
| `mercado_financeiro` | Selic and interest rates, credit, banks, Ibovespa, the dollar/exchange rate, inflation and expectations, public debt, financial regulation |
| `mercado_trabalho` | employment, unemployment, wages, formal jobs (Caged, PNAD), layoffs, labour law |
| `servicos` | private services activity: retail and commerce, tourism, transport, telecoms, restaurants and events |

Rules:
- **R1. Incidental setting is not relevance.** Crime, accidents, politics, health, culture and
  sports that merely occur at a business, or mention "indústria"/"trabalho", are `none`.
  (e.g. a cashier robbed at work, a nurse found dead at her workplace.)
- **R2. Local news counts** when it is about economic activity: a municipal job programme for the
  unemployed is `mercado_trabalho`; a regional industrial fair that fills hotels is `servicos`
  or `industria` by main subject.
- **R3. Public-service delivery is not `servicos`.** Document-issuing task forces, free health
  or psychological support, public campaigns are `none`. `servicos` is private-sector activity.
- **R4. One topic only.** Choose by main subject. If still tied, use the order: the thing the
  headline is about wins; mark `ambiguous`.

**DECIDED D1: foreign news.** E.g. "Atividade da indústria dos EUA tem leve desaceleração",
"Índices europeus fecham em alta".
*Decision (accepted):* `none` unless Brazil is explicitly in the text (Brazilian exports, firms,
market, or stated effect on Brazil). The study forecasts Brazilian activity, and a rule keyed
on mentioning Brazil is easy to apply consistently. The model currently breaks this rule
(it labelled the US industry item relevant), which is exactly what this set should measure.

**DECIDED D2: global commodities and macro (oil, sugar, China).**
*Decision (accepted):* relevant only with an explicit Brazil link (Petrobras, Brazilian fuel prices,
Brazilian producers or exports). Global-only price news is `none`.

**DECIDED D3: personal finance, entrepreneurship, support programmes.** E.g. Sebrae financial
help for flood victims; entrepreneurs networking.
*Decision (accepted):* relevant only if it is about credit, jobs or output in the economy (Sebrae
credit → `mercado_financeiro`). Networking, self-help and business-advice pieces are `none`.

## 2. Sentiment (3-class, economic not emotional)
Label what the text implies for **Brazilian economic activity in the article's topic**:
- **positive**: growth, rising output or sales, hiring, investment, easing costs, good outlook.
- **negative**: contraction, falling output, layoffs, rising unemployment or costs, crisis.
- **neutral**: no clear direction, purely descriptive, or genuinely balanced.

Rules:
- **S1.** Judge the economic content, not the prose tone. A cheerful piece about a closure is negative.
- **S2. Mixed** news: pick the direction of the main finding. If it is truly 50/50, `neutral`
  and mark `ambiguous`.
- **S3. Good for one actor, bad for the economy** (or vice versa): judge for the aggregate economy.
- **S4. Outlook and forecasts count** when they are about the economy's direction.
- **S5.** Irrelevant articles get no sentiment label (leave blank). Sentiment is scored on
  relevant articles only.

**DECIDED D4: market moves, e.g. "Dólar fecha em alta em dia de poucas negociações".**
*Decision (accepted):* purely descriptive daily wraps are `neutral`; label positive or negative only
when the text says why the move is good or bad news for the Brazilian economy. Otherwise the
label would measure price direction (a dollar rise is not inherently bad), not sentiment. The
model scored this case -0.3, so it applies the opposite convention.

## 3. Economic policy uncertainty (EPU)
`epu = y` only if **both** hold:
1. the article is about **Brazilian economic policy** (fiscal, tax, monetary, exchange-rate,
   regulatory, trade, labour-law or industrial policy); **and**
2. it presents that policy as **undecided, contested, shifting or unclear in its effect**, so
   firms or households are left unsure.

Not EPU:
- a decided, announced policy with clear effects (e.g. "Receita reduzirá IPI para carros com
  peças nacionais"): that is certainty, not uncertainty;
- political instability or scandal with no economic-policy angle;
- foreign political or policy uncertainty (this is the Peru case: relevant-sounding
  instability, but not Brazilian, so `epu = n` and the article is `none`);
- "incerteza" about weather, demand or markets with no policy link.

Illustrative positives (invented, not from the corpus): "Governo ainda não decidiu se prorroga
a desoneração da folha; empresários aguardam"; "Mudança nas regras do ICMS gera dúvida entre
exportadores".

**An irrelevant article is never EPU** (the pipeline enforces this, so label it `n`).

## 4. Process and quality control
- One labeller. To estimate your own consistency, **re-label a random 10% after about a week**,
  blind to your first labels, and we report intra-annotator κ.
- Ambiguous cases: give your best label, set `ambiguous = y`, add a one-line note. Log them in
  the table below. They are reported, not hidden.
- Split: 100 articles **dev** (used to tune the prompt and choose few-shot examples) and 200
  **test** (frozen, never used for tuning).
- Sampling (to be implemented, not yet done): stratified by topic × year; relevant and
  irrelevant both kept; **EPU is oversampled using the original thesis' policy-and-uncertainty
  keyword hits**, which are independent of the LLM, so measured EPU recall is not biased toward
  what the model already flags. A random 300 would hold only a handful of EPU positives.

## Ambiguous cases log

| Article id | Issue | Decision |
|---|---|---|
| mercado_financeiro-32327 | Peru PM resigns; real policy instability but foreign | `none`, EPU n |
| industria-8034 | US industrial activity slows | D1 |
| mercado_financeiro-25850 | Oil up 4% | D2 |
| mercado_financeiro-9793 | Dollar closes higher | D4 |

## Worked examples
Eight invented examples (none from the gold set) are in the first sheet of
`data/gold/labels.xlsx` ("How to label"), generated from `eval/make_label_sheet.py`. They cover:
a clear negative industry item, incidental crime (`none`), foreign news (D1), a descriptive
dollar wrap (D4, neutral), a decided policy (EPU `n`), a contested policy (EPU `y`), a clear
positive agribusiness item, and an ambiguous wage-floor item (S3, `ambiguous = y`).

## Open decisions raised by the first labelling pass (Claude's provisional labels)
Claude labelled all 300 items blind to the model, from this guide. Where the guide was silent,
one call was made and applied consistently. Each is a class you can overrule once; the item ids
are in `data/gold/claude_labels.csv` (notes mention the tag). Counts are approximate.

| # | Class | Items | What was done | Alternative |
|---|---|---|---|---|
| D5 | Job-vacancy lists, training-programme and civil-service-exam listings | ~16 | relevant, `mercado_trabalho`, neutral | mildly positive (hiring), or not relevant |
| D6 | Macro aggregates and trade balance (GDP, trade surplus, consumer confidence, fiscal) | ~16 | forced into the nearest of the five topics (trade → industria, GDP/inflation/fiscal → mercado_financeiro, confidence → servicos) | add a sixth topic `macro`, or treat as not relevant; forcing makes topic accuracy partly arbitrary |
| D7 | Campaign promises and political speeches about the economy | ~6 | not relevant | relevant when they concern concrete economic policy |
| R3 | Public works and public-service delivery | ~10 | not relevant | relevant when there is an explicit economic purpose (done for g129, g269) |
| D3 | Entrepreneurship profiles and business-advice features | ~4 | not relevant | relevant |
| D8 | M&A and ownership changes | ~5 | neutral sentiment | positive (capital inflow) |
| D9 | Trade-fair and event announcements | ~8 | relevant, neutral/positive, `none` for parades and festivals | not relevant |
| D10 | EPU from another country's policy that directly targets Brazil (g046, US steel tariffs) | 1 | EPU `y` | `n` per the "foreign-only" rule |
| D11 | Regulated utility tariff adjustments | ~5 | `industria`, EPU `n` | `mercado_financeiro` (administered prices) |
| D12 | Historical retrospectives (g175) | 1 | not relevant | relevant |

Review order that costs the least time: (1) decide D5-D12 once; (2) read all 31 EPU = `y` and the
66 not-relevant items; (3) spot-check 15 `high`-confidence items (if you overrule several, my
confidence is miscalibrated).

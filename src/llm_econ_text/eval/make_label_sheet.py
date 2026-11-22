"""Build the fill-in workbook data/gold/labels.xlsx from the blind file data/gold/to_label.csv.

Sheet "How to label": the rules and worked examples (invented, none from the gold set).
Sheet "Labels": one row per article, dropdowns on every label column.
Refuses to overwrite an existing workbook (it may hold hours of labels) unless --force.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

TOPICS = "industria,agronegocio,mercado_financeiro,mercado_trabalho,servicos"
COLUMNS = ["item", "headline", "lead", "relevant", "topic", "sentiment", "epu", "ambiguous", "note"]

STEPS = [
    "1. Read ONLY the headline and lead. Do not open the article or search for it.",
    ("2. relevant: y if the main subject is Brazilian economic activity in one of the five topics, "
    "else n. If n, leave topic, sentiment and epu EMPTY."),
    "3. topic: pick the one topic the article is mainly about (dropdown).",
    ("4. sentiment: negative / neutral / positive for the Brazilian economy in that topic. "
    "Judge the economic content, not the tone of the prose."),
    ("5. epu: y only if the article is about Brazilian economic POLICY and presents it as "
    "undecided, contested or unclear. Otherwise n. An irrelevant article is never EPU."),
    ("6. ambiguous: y if you hesitated between two labels. Give your best label anyway and say "
    "why in note (one line)."),
    ("7. Label blind: do not look at any model output. Save as .xlsx (keep this file name). "
    "Do not delete, rename or reorder columns. You can label in several sessions."),
]

EXAMPLES = [
    # headline, lead, relevant, topic, sentiment, epu, ambiguous, why
    ("Produção industrial recua 2,1% em março e é a maior queda desde 2020",
     ("O IBGE informou que a produção da indústria brasileira caiu em março, com queda em 18 "
     "dos 26 ramos pesquisados."), "y", "industria", "negative", "n", "n",
     "Brazilian industrial output falling."),
    ("Gerente de loja é rendido durante assalto em Londrina",
     ("Dois homens armados renderam o gerente de uma loja de calçados na noite desta terça-feira. "
     "A polícia procura os suspeitos."), "n", "", "", "", "n",
     "Incidental setting (R1): crime that happens at a business is not economic news."),
    ("Produção fabril da Alemanha recua em maio, informa instituto de estatística",
     "O índice de produção industrial alemão caiu 0,8% no mês, segundo o órgão oficial do país.",
     "n", "", "", "", "n", "Foreign news with no Brazil link (D1)."),
    ("Ibovespa fecha estável em sessão de baixo volume",
     "O principal índice da bolsa brasileira terminou o dia praticamente estável, com giro abaixo da média.",
     "y", "mercado_financeiro", "neutral", "n", "n",
     "Descriptive market wrap, no stated good or bad news (D4): neutral."),
    ("Governo zera imposto de importação de máquinas agrícolas a partir de março",
     "A medida, já publicada no Diário Oficial, vale por um ano e deve reduzir o custo de equipamentos rurais.",
     "y", "agronegocio", "positive", "n", "n",
     "A decided, announced policy is certainty, not uncertainty: epu = n."),
    ("Reforma tributária: governo e Congresso divergem sobre alíquota; indústria teme custos",
     "Sem acordo sobre o texto, empresários dizem não saber como planejar investimentos.",
     "y", "industria", "negative", "y", "n",
     "Policy is contested and firms are unsure: epu = y."),
    ("Safra de soja deve crescer 8% e puxar exportações, diz Conab",
     "A estimativa considera área plantada maior e clima favorável nas principais regiões.",
     "y", "agronegocio", "positive", "n", "n", "Clear positive agribusiness news."),
    ("Salário mínimo regional sobe 7% no Paraná",
     "O novo piso vale para todas as categorias e entra em vigor no próximo mês.",
     "y", "mercado_trabalho", "positive", "n", "y",
     ("Good for workers, a cost for employers (S3): judge for the aggregate economy, mark "
     "ambiguous and explain in note.")),
]


def build(src: Path, out: Path, prefill: pd.DataFrame | None = None) -> None:
    """`prefill`: optional frame keyed by item with label columns (and `confidence`) to pre-fill."""
    df = pd.read_csv(src, dtype=str, keep_default_na=False)
    wb = Workbook()
    guide = wb.active
    guide.title = "How to label"
    guide.column_dimensions["A"].width = 42
    guide.column_dimensions["B"].width = 60
    for c in "CDEFGH":
        guide.column_dimensions[c].width = 14
    guide.column_dimensions["I"].width = 60
    guide["A1"] = "How to label (full rules: docs/labeling_guide.md)"
    guide["A1"].font = Font(bold=True, size=14)
    for i, step in enumerate(STEPS, start=3):
        guide.cell(row=i, column=1, value=step)
        guide.merge_cells(start_row=i, start_column=1, end_row=i, end_column=9)
        guide.cell(row=i, column=1).alignment = Alignment(wrap_text=True, vertical="top")
        guide.row_dimensions[i].height = 32
    r0 = 3 + len(STEPS) + 1
    guide.cell(row=r0, column=1, value="Worked examples (all invented: none is a corpus headline or in your set)").font = Font(
        bold=True, size=12)
    hdr = ["headline", "lead", "relevant", "topic", "sentiment", "epu", "ambiguous", "(blank)",
           "why"]
    for j, h in enumerate(hdr, start=1):
        c = guide.cell(row=r0 + 1, column=j, value=h)
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="DDDDDD")
    for i, (h, ld, rel, top, sen, epu, amb, why) in enumerate(EXAMPLES, start=r0 + 2):
        for j, v in enumerate([h, ld, rel, top, sen, epu, amb, "", why], start=1):
            c = guide.cell(row=i, column=j, value=v)
            c.alignment = Alignment(wrap_text=True, vertical="top")

    ws = wb.create_sheet("Labels")
    cols = [*COLUMNS, "confidence"] if prefill is not None else COLUMNS
    ws.append(cols)
    pre = prefill.set_index("item") if prefill is not None else None
    for r in df.itertuples():
        row = [r.item, r.headline, r.lead] + [None] * (len(cols) - 3)
        if pre is not None and r.item in pre.index:
            for j, c in enumerate(cols[3:], start=3):
                v = pre.loc[r.item].get(c, "")
                row[j] = None if (pd.isna(v) or v == "") else v
        ws.append(row)
    widths = {"A": 7, "B": 42, "C": 90, "D": 10, "E": 20, "F": 11, "G": 7, "H": 11, "I": 50, "J": 11}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="DDDDDD")
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "D2"
    n = ws.max_row
    validations = {"D": "y,n", "E": TOPICS, "F": "negative,neutral,positive", "G": "y,n",
                   "H": "y,n"}
    if prefill is not None:
        validations["J"] = "high,medium,low"
    for col, options in validations.items():
        dv = DataValidation(type="list", formula1=f'"{options}"', allow_blank=True,
                            showErrorMessage=True,
                            errorTitle="Invalid value", error=f"Choose one of: {options}")
        ws.add_data_validation(dv)
        dv.add(f"{col}2:{col}{n}")
    wb.active = 0
    wb.save(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/gold/to_label.csv")
    ap.add_argument("--out", default="data/gold/labels.xlsx")
    ap.add_argument("--force", action="store_true", help="overwrite an existing workbook")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} already exists and may contain your labels. Use --force to "
                         "overwrite.")
    build(Path(args.src), out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

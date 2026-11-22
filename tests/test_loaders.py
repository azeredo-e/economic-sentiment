from llm_econ_text.data.loaders import extract_lead, headline_from_url

URL = (
    "https://g1.globo.com/busca/click?q=Ind%C3%BAstria&p=1&u=https%3A%2F%2Fg1.globo.com%2F"
    "economia-e-negocios%2Fnoticia%2F2010%2F05%2Fcom-vendas-em-alta-fabricantes-de-tv.html&syn=False"
)


def test_headline_from_search_click_url():
    assert headline_from_url(URL) == "Com vendas em alta fabricantes de tv"


def test_headline_from_plain_url():
    assert headline_from_url("https://g1.globo.com/a/noticia/2020/01/pib-cresce.ghtml") == "Pib cresce"


def test_extract_lead_skips_metadata():
    text = (
        "01/05/2010 21h02\n- Atualizado em\n01/05/2010 21h02\nDo G1, com informações do JN\n"
        "Faltam 41 dias para a Copa.\nSegundo parágrafo.\n"
    )
    assert extract_lead(text) == "Faltam 41 dias para a Copa. Segundo parágrafo."


def test_extract_lead_caps_length():
    text = "Primeiro parágrafo. " * 5 + "\n" + "Segundo " * 200
    assert len(extract_lead(text, max_chars=200)) <= 200


def test_lead_new_format_drops_byline_stanza_and_caption():
    text = (
        "fique por\ndentro\nPor G1 SC\n16/02/2018 16h59\nAtualizado\n23/02/2018\n"
        "Legenda de uma fábrica em Joinville — Foto: Fulano/Reuters\n"
        "A produção industrial cresceu em abril, segundo dados divulgados nesta terça.\n"
        "Segundo parágrafo com mais detalhes."
    )
    assert extract_lead(text) == (
        "A produção industrial cresceu em abril, segundo dados divulgados nesta terça. "
        "Segundo parágrafo com mais detalhes."
    )


def test_lead_rejoins_hyperlink_fragments_without_eating_them():
    text = ("Por g1\n01/01/2021 10h00\nAtualizado\n01/01/2021\n"
            "A\nChina\n,\npaís com muitos bilionários\n, tem\num novo magnata.")
    assert extract_lead(text) == "A China, país com muitos bilionários, tem um novo magnata."


def test_lead_edicao_do_dia_header_with_reporter_and_place():
    text = ("Edição do dia 01/11/2013\n02/11/2013 00h12\n- Atualizado em\n02/11/2013 01h20\n"
            "Fulana de Tal\nBrasília, DF\n"
            "O governo anunciou novas medidas fiscais nesta sexta-feira em Brasília.")
    assert extract_lead(text).startswith("O governo anunciou")


def test_lead_without_stanza_falls_back_to_prefix_rules():
    text = "Por Fulano\n10/10/2020 10h10\nTexto normal de lead com conteúdo suficiente."
    assert extract_lead(text) == "Texto normal de lead com conteúdo suficiente."


def test_lead_cut_never_lands_mid_word():
    para = " ".join(["Esta é uma frase completa sobre a economia brasileira."] * 30)
    out = extract_lead("Atualizado\n01/01/2021\n" + para, max_chars=200)
    assert len(out) <= 200 and out.endswith(".")
    nospace = extract_lead("Atualizado\n01/01/2021\n" + "palavra " * 200, max_chars=100)
    assert nospace.endswith("…") and not nospace.endswith("palav…")


def test_lead_drops_promo_lines_but_keeps_real_sentences():
    text = "Atualizado\n01/01/2021\nVeja também: outras notícias\nO IBGE divulgou o índice. Veja os dados."
    assert extract_lead(text) == "O IBGE divulgou o índice. Veja os dados."


def test_lead_skips_long_outlet_and_sponsor_bylines():
    a = "Atualizado\n01/01/2021\nDo G1 MA com informações da TV Mirante\nO Mirante Rural mostrou a safra."
    b = "Atualizado\n01/01/2021\npor Governo de Rondônia conteúdo de responsabilidade do anunciante\nOs organizadores abriram as inscrições."
    assert extract_lead(a) == "O Mirante Rural mostrou a safra."
    assert extract_lead(b) == "Os organizadores abriram as inscrições."
    keep = "Atualizado\n01/01/2021\nDe acordo com o IBGE, a produção caiu 2% em março."
    assert extract_lead(keep).startswith("De acordo com o IBGE")  # real sentence is not a byline


def test_lead_skips_sponsored_content_line():
    t = ("Atualizado\n01/01/2021\npor CCR Rodonorte conteúdo de responsabilidade do anunciante\n"
         "A maior parte da produção agrícola da região cresceu.")
    assert extract_lead(t) == "A maior parte da produção agrícola da região cresceu."

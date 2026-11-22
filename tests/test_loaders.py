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

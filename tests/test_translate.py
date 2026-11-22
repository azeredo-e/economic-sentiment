import json
from types import SimpleNamespace

from llm_econ_text.llm.translate import Translator


class FakeClient:
    def __init__(self):
        self.calls = 0
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls += 1
        payload = json.loads(kw["messages"][0]["content"])
        out = {"headline_en": "EN " + payload["headline"], "lead_en": "EN " + payload["lead"] if payload["lead"] else ""}
        usage = SimpleNamespace(input_tokens=500, cache_creation_input_tokens=0,
                                cache_read_input_tokens=0, output_tokens=200)
        return SimpleNamespace(usage=usage, content=[SimpleNamespace(type="text", text=json.dumps(out))])


def make(tmp_path, client):
    return Translator("sys", "translate_v1", db_path=tmp_path / "c.sqlite", client=client)


def test_translate_caches_and_logs_cost(tmp_path):
    fake = FakeClient()
    t = make(tmp_path, fake)
    a = t.translate("a1", "titulo", "lide")
    b = t.translate("a1", "titulo", "lide")
    assert a == b and a.headline_en == "EN titulo" and fake.calls == 1
    assert t.s.total_spent() > 0
    assert t.translate("a1", "titulo", "OUTRO lide").lead_en == "EN OUTRO lide"  # input changed
    assert fake.calls == 2


def test_empty_lead_stays_empty(tmp_path):
    assert make(tmp_path, FakeClient()).translate("a2", "so titulo", "").lead_en == ""


def test_translation_rows_do_not_pollute_label_cache(tmp_path):
    t = make(tmp_path, FakeClient())
    t.translate("a1", "titulo", "lide")
    assert t.s.get_label("a1", "titulo", "lide") is None  # status 'translated' is not a label

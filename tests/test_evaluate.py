import numpy as np
import pandas as pd

from llm_econ_text.eval import evaluate


def synthetic(tmp_path, n=120, noise=0.15, seed=0):
    rng = np.random.default_rng(seed)
    topics = ["industria", "agronegocio", "mercado_financeiro", "mercado_trabalho", "servicos"]
    sent = np.array(["negative", "neutral", "positive"])
    rows, design = [], []
    for i in range(n):
        item = f"g{i + 1:03d}"
        rel = rng.random() < 0.6
        topic = rng.choice(topics) if rel else "none"
        s = rng.choice(sent) if rel else None
        epu = bool(rel and rng.random() < 0.15)
        flip = lambda v: (not v) if rng.random() < noise else v
        p_rel = flip(rel)
        p_topic = topic if p_rel and rel else (rng.choice(topics) if p_rel else "none")
        score = {"negative": -0.5, "neutral": 0.0, "positive": 0.5}[s] if s else 0.0
        design.append({
            "item": item, "article_id": f"x{i}", "url": "u", "date": "2015-01-01",
            "source_topic": "industria", "stratum": ["S1", "S2", "S3", "S4"][i % 4],
            "strat_topic": topic, "weight": float(rng.integers(1, 50)), "w_pool": 1.0,
            "N_cell": 1, "n_cell_gold": 1, "split": "dev" if i % 3 == 0 else "test",
            "relevant": p_rel, "llm_topic": p_topic,
            "llm_sentiment": score if p_rel else 0.0, "epu": flip(epu) and p_rel,
            "llm_confidence": 0.8, "thesis_hit": False,
        })
        rows.append({"item": item, "headline": "h", "lead": "l",
                     "relevant": "y" if rel else "n", "topic": topic if rel else "",
                     "sentiment": s or "", "epu": ("y" if epu else "n") if rel else "",
                     "ambiguous": "", "note": ""})
    g, d = tmp_path / "labels.csv", tmp_path / "design.csv"
    pd.DataFrame(rows).to_csv(g, index=False)
    pd.DataFrame(design).to_csv(d, index=False)
    return str(g), str(d)


def test_end_to_end_report(tmp_path, capsys):
    g, d = synthetic(tmp_path)
    evaluate.main(["--gold", g, "--design", d, "--name", "t", "--out", str(tmp_path / "out"),
                   "--n-boot", "50"])
    md = (tmp_path / "out" / "eval_t.md").read_text()
    for needle in ["### Relevance", "### Topic", "### Sentiment", "### EPU",
                   "Strata diagnostics", "Split: test"]:
        assert needle in md
    assert (tmp_path / "out" / "eval_t.json").exists()


def test_partial_labels_only_score_valid_rows(tmp_path):
    g, d = synthetic(tmp_path)
    df = pd.read_csv(g, dtype=str, keep_default_na=False)
    df.loc[60:, ["relevant", "topic", "sentiment", "epu"]] = ""  # half still pending
    df.to_csv(g, index=False)
    evaluate.main(["--gold", g, "--design", d, "--name", "p", "--out", str(tmp_path / "out"),
                   "--n-boot", "20"])
    md = (tmp_path / "out" / "eval_p.md").read_text()
    assert "gold labelled: 60 of 120" in md and "provisional" in md


def test_intra_annotator_kappa_perfect_on_identical(tmp_path):
    g, _ = synthetic(tmp_path)
    gold, _, _ = evaluate.load_gold(g)
    k = evaluate.intra_annotator(gold, gold)
    assert k["g_relevant"] == 1.0

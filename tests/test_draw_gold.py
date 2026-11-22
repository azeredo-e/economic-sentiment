import numpy as np
import pandas as pd

from llm_econ_text.eval.draw_gold import draw, round_robin_take


def fake_pool(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    p = pd.DataFrame({
        "article_id": [f"a{i}" for i in range(n)],
        "source_topic": rng.choice(["industria", "agronegocio"], n),
        "llm_topic": rng.choice(["industria", "agronegocio", "none"], n, p=[.3, .2, .5]),
        "year": rng.integers(2010, 2024, n),
        "thesis_hit": rng.random(n) < 0.3,
        "epu": rng.random(n) < 0.02,
        "w_pool": 2.0,
    })
    p["relevant"] = p.llm_topic != "none"
    p["epu"] = p.epu & p.relevant
    return p


def test_round_robin_spreads_over_cells():
    df = pd.DataFrame({"c": ["a"] * 50 + ["b"] * 50, "x": range(100)})
    out = round_robin_take(df, 10, ["c"], seed=1)
    assert out.c.value_counts().to_dict() == {"a": 5, "b": 5}


def test_draw_sizes_strata_and_split():
    g = draw(fake_pool(), total=300, seed=42)
    assert len(g) == 300 and g.article_id.is_unique
    assert (g.stratum == "S3_llm_irrelevant").sum() == 60
    assert (g.stratum == "S4_thesis_hit").sum() == 50
    assert (g.stratum == "S2_llm_epu").sum() <= 40
    assert abs((g.split == "dev").sum() - 100) <= 5
    assert not g[g.stratum == "S3_llm_irrelevant"].relevant.any()
    assert (g.weight > 0).all()


def test_weights_recover_population_even_with_undrawn_cells():
    """Sparse regime like the real draw: far fewer draws than topic x year cells."""
    from llm_econ_text.eval.draw_gold import assign_strata

    p = fake_pool(n=1500, seed=3)
    g = draw(p, total=120, seed=7, quotas={"S2_llm_epu": 10, "S4_thesis_hit": 20,
                                           "S3_llm_irrelevant": 15})
    assert (g.weight_raw.groupby(g.stratum).sum() < assign_strata(p).groupby("stratum").w_pool.sum()
            ).any()  # the raw cell weights do under-cover somewhere
    pop = assign_strata(p).groupby("stratum").w_pool.sum()
    est = g.groupby("stratum").weight.sum()
    for s in est.index:
        assert abs(est[s] - pop[s]) < 1e-6 * pop[s]


def test_refresh_gold_inputs_keeps_items_and_order(tmp_path, monkeypatch):
    from llm_econ_text.eval import refresh_gold_inputs as r

    pool = tmp_path / "pool.csv"
    pd.DataFrame({"article_id": ["a1", "a2", "a3"], "headline": ["h1", "h2", "h3"],
                  "lead": ["NEW1", "NEW2", "NEW3"]}).to_csv(pool, index=False)
    design = tmp_path / "design.csv"
    pd.DataFrame({"item": ["g002", "g001"], "article_id": ["a3", "a1"]}).to_csv(design, index=False)
    blind = tmp_path / "to_label.csv"
    pd.DataFrame({"item": ["g002", "g001"], "headline": ["x", "y"], "lead": ["OLD", "OLD"]}
                 ).to_csv(blind, index=False)
    monkeypatch.setattr(r, "POOL_PATH", pool)
    out = r.refresh(str(design), str(blind))
    assert list(out.item) == ["g002", "g001"] and list(out.lead) == ["NEW3", "NEW1"]
    assert (tmp_path / "to_label_old_leads.csv").exists()


def test_refresh_keeps_articles_with_empty_lead(tmp_path, monkeypatch):
    from llm_econ_text.eval import refresh_gold_inputs as r

    pool = tmp_path / "pool.csv"
    pd.DataFrame({"article_id": ["a1"], "headline": ["h1"], "lead": [""]}).to_csv(pool, index=False)
    design = tmp_path / "design.csv"
    pd.DataFrame({"item": ["g001"], "article_id": ["a1"]}).to_csv(design, index=False)
    monkeypatch.setattr(r, "POOL_PATH", pool)
    out = r.refresh(str(design), str(tmp_path / "to_label.csv"))
    assert list(out.lead) == [""]

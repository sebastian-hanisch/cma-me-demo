"""AppTest-Rauchtests: Voreinstellung, jedes Preset, alle Stufen, Längen, Verfahren und Schrittweiten, Randwerte, lokale Regler, Würfel-Knopf, Permalink-Grenzen, Abschnitte, Footer."""

import random
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import cme_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=600)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_default_run_has_no_exception_and_shows_the_reference_values():
    at = _run()
    _ok(at)
    assert _metric(at, "Kürzester Weg") == "17 Zellen" and _metric(at, "Fallen (lokale Minima)") == "0" and _metric(at, "Freie Zellen") == "484"
    assert _metric(at, "MAP-Elites: belegte Zellen").endswith("von 144") and _metric(at, "CMA: belegte Zellen").endswith("von 144") and _metric(at, "CMA: Neustarts der Emitter").isdigit()
    assert len(at.get("plotly_chart")) == 10


def test_all_sections_are_present():
    at = _run()
    assert [s.value for s in at.subheader] == ["📐 Ein Lauf: MAP-Elites gegen ein CMA-Verfahren", "🔬 Wann gewinnt CMA? Die Studie über 20 Läufe", "🔬 Wie wichtig ist die Anfangsschrittweite σ₀?", "🚧 Wo die Annahmen enden"]
    assert any(m.value.startswith("## 🔗 Wie täuschend ist diese Stufe?") for m in at.markdown)
    assert any("Diese Demo ist Teil des Portfolios" in c.value for c in at.caption)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    assert (at.session_state["level_select"], at.session_state["gens_slider"], at.session_state["steps_select"], at.session_state["sigma_select"], at.session_state["method_select"]) == \
        (p["level"], p["gens"], p["steps"], p["sigma"], p["method"])
    assert at.metric


@pytest.mark.parametrize("level", list(C.LEVELS))
def test_every_level_runs(level):
    at = _run(level_select=level, gens_slider=10, steps_select=20)
    _ok(at)
    assert _metric(at, "MAP-Elites: belegte Zellen").endswith("von 144")


@pytest.mark.parametrize("steps", C.STEP_OPTIONS)
def test_every_genome_length_runs(steps):
    at = _run(steps_select=steps, gens_slider=20)
    _ok(at)
    assert len(at.get("plotly_chart")) == 10 and _metric(at, "MAP-Elites: QD-Score")


@pytest.mark.parametrize("method", C.CMA_METHODS)
def test_every_cma_method_runs(method):
    at = _run(method_select=method, gens_slider=20, steps_select=20)
    _ok(at)
    assert _metric(at, "CMA: QD-Score") and len(at.get("plotly_chart")) == 10


@pytest.mark.parametrize("sigma", C.SIGMAS)
def test_every_sigma_runs(sigma):
    _ok(_run(sigma_select=sigma, gens_slider=20, steps_select=20))


def test_the_unreachable_goal_is_explained_for_short_genomes_only():
    short = _run(steps_select=20, gens_slider=10)
    assert any("Das Ziel ist hier unerreichbar" in c.value for c in short.caption) and _metric(short, "MAP-Elites: Evaluationen bis zum Erfolg") == "–"
    longer = _run(steps_select=140, gens_slider=10)
    assert not any("Das Ziel ist hier unerreichbar" in c.value for c in longer.caption)


def test_the_live_info_names_the_scores_and_the_restarts():
    at = _run(gens_slider=30, steps_select=20)
    text = next(i.value for i in at.info if i.value.startswith("Nach 30 Generationen hat das CMA-Verfahren"))
    assert "MAP-Elites:" in text and "neu gestartet" in text and "%" in text


def test_the_solvability_line_names_the_steps_used():
    assert any("136 von 140 Schritten benutzt" in c.value for c in _run(level_select=4).caption)


@pytest.mark.parametrize("kw", [dict(gens_slider=10), dict(gens_slider=200, steps_select=10), dict(level_select=3, gens_slider=40, steps_select=140), dict(level_select=4, steps_select=10, gens_slider=10, seed_input=0),
                                dict(seed_input=999999, gens_slider=10, method_select="cmame", sigma_select=0.4, steps_select=80)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_the_gain_table_has_one_row_per_cma_method_and_one_column_per_case():
    at = _run()
    table = next(m.value for m in at.markdown if m.value.startswith("| Gewinn des QD-Scores gegenüber MAP-Elites |"))
    assert table.count("\n| ") == len(C.CMA_METHODS) and table.splitlines()[0].count("|") == len(C.CASES) + 2
    assert all(C.CASE_LABELS[c] in table.splitlines()[0] for c in C.CASES)


def test_the_study_regulator_switches_the_case_and_the_detail_table():
    at = _run(study_case_select="d140")
    _ok(at)
    table = next(m.value for m in at.markdown if m.value.startswith(f"**{C.CASE_LABELS['d140']}**"))
    assert table.count("\n| ") == len(C.METHODS) + 1 and len(at.get("plotly_chart")) == 10               # Kopfzeile + eine Zeile je Verfahren (die Trennzeile beginnt mit |---)


def test_the_study_info_names_the_gain_at_few_dimensions_and_at_the_native_dimension():
    at = _run()
    text = next(i.value for i in at.info if i.value.startswith("**Wenige Dimensionen (20 Schritte, 40 Dimensionen):**"))
    assert "CMA-ME +" in text and "280 Dimensionen" in text and "Der Vorsprung schwindet mit der Dimension" in text


def test_the_sigma_info_names_the_best_and_the_extreme_values():
    at = _run()
    text = next(i.value for i in at.info if i.value.startswith("Bei 40 Schritten liegt der beste Wert für CMA-ME"))
    assert "σ₀ = 0.05 fallen sie auf" in text and "MAP-Elites:" in text


def test_dice_button_changes_the_seed_and_the_run(monkeypatch):
    """Der Würfel zieht sonst einen unseeded Zufalls-Seed; deshalb ist der gewürfelte Seed im Test fest."""
    monkeypatch.setattr(random, "randint", lambda a, b: 508145)
    at = _run(steps_select=20)
    old_seed = at.session_state["seed_input"]
    old = (_metric(at, "MAP-Elites: QD-Score"), _metric(at, "CMA: QD-Score"))
    next(b for b in at.button if b.label == "🎲 Neuen Lauf würfeln").click().run()
    _ok(at)
    assert at.session_state["seed_input"] == 508145 != old_seed
    assert (_metric(at, "MAP-Elites: QD-Score"), _metric(at, "CMA: QD-Score")) != old


def test_permalink_values_are_clamped_and_snapped():
    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["level"] = "9"
    at.query_params["gens"] = "104"
    at.query_params["steps"] = "50"
    at.query_params["sigma"] = "0.3"
    at.query_params["method"] = "mae01"
    at.run()
    _ok(at)
    assert at.session_state["level_select"] == 4 and at.session_state["gens_slider"] == 100 and at.session_state["steps_select"] == 40 and at.session_state["sigma_select"] == 0.2 and at.session_state["method_select"] == "mae01"


def test_permalink_ignores_garbage():
    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["level"] = "viele"
    at.query_params["seed"] = "nan"
    at.query_params["method"] = "me"
    at.query_params["sigma"] = "breit"
    at.query_params["steps"] = "0"
    at.run()
    _ok(at)
    assert at.session_state["level_select"] == C.DEFAULT_LEVEL and at.session_state["seed_input"] == C.DEFAULT_SEED and at.session_state["method_select"] == C.DEFAULT_METHOD
    assert at.session_state["sigma_select"] == C.DEFAULT_SIGMA and at.session_state["steps_select"] == C.STEP_OPTIONS[0]


def test_no_sentence_wide_comma_replacement_in_the_app_source():
    """Regressionsschutz: `.replace(",", ".")` auf einem ganzen (verketteten) Satz macht aus Kommas im Fließtext Punkte; Tausender nur über `fmt_int`."""
    assert '.replace(",", ".")' not in Path(APP).read_text(encoding="utf-8")

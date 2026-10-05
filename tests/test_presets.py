"""Presets: Vollständigkeit, gültige Werte, Permalink-Konstanten, Formatierer."""

import pytest

import cme_constants as C
import cme_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_ORDER) and len(C.PRESETS) == 4
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS) and C.PRESET_HELP[name]


def test_preset_values_are_valid_and_match_the_setting_specs():
    for preset in C.PRESETS.values():
        assert preset["level"] in C.LEVELS and C.GENS_MIN <= preset["gens"] <= C.GENS_MAX and P.snap_gens(preset["gens"]) == preset["gens"]
        assert preset["steps"] in C.STEP_OPTIONS and preset["sigma"] in C.SIGMAS and preset["method"] in C.CMA_METHODS
        for key, state_key in P.PRESET_KEYS.items():
            P.SETTING_SPECS[state_key].caster(preset[key])


def test_preset_names_state_the_values_they_set():
    assert C.PRESETS["Wenige Dimensionen"]["steps"] == 20 and C.PRESETS["Native Dimension"]["steps"] == 140 == C.STEPS
    assert C.PRESETS["Falle"]["level"] == 3 and C.PRESETS["Falle"]["method"] == "cmame" and C.PRESETS["Zu kleine Schrittweite"]["sigma"] == min(C.SIGMAS)


def test_default_preset_equals_the_default_settings():
    p = C.PRESETS["Wenige Dimensionen"]
    assert (p["level"], p["gens"], p["seed"], p["sigma"], p["method"]) == (C.DEFAULT_LEVEL, C.DEFAULT_GENS, C.DEFAULT_SEED, C.DEFAULT_SIGMA, C.DEFAULT_METHOD)


def test_bounds_and_url_params():
    assert P.bounds("seed_input") == (0, C.SEED_MAX) and P.bounds("gens_slider") == (10, 200) and P.bounds("level_select") == (0, 4) and P.bounds("steps_select") == (10, 140) and P.bounds("sigma_select") == (0.05, 0.4)
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS) == 6


@pytest.mark.parametrize("value,expected", [(0, 10), (14, 10), (16, 20), (104, 100), (106, 110), (999, 200)])
def test_generations_snap_to_the_step_from_the_lower_bound(value, expected):
    assert P.snap_gens(value) == expected


@pytest.mark.parametrize("value,expected", [(10, 10), (30, 20), (50, 40), (60, 40), (61, 80), (140, 140)])
def test_step_counts_snap_to_the_nearest_option_and_ties_go_down(value, expected):
    assert P.snap_to_option("steps_select", value) == expected


@pytest.mark.parametrize("value,expected", [(0.05, 0.05), (0.07, 0.05), (0.08, 0.1), (0.3, 0.2), (0.4, 0.4)])
def test_sigmas_snap_to_the_nearest_option(value, expected):
    assert P.snap_to_option("sigma_select", value) == expected


@pytest.mark.parametrize("value", C.CMA_METHODS)
def test_method_caster_accepts_the_cma_methods(value):
    assert P.SETTING_SPECS["method_select"].caster(value) == value


@pytest.mark.parametrize("value", ["me", "map-elites", "", 3, None, ["cmame"]])
def test_method_caster_refuses_everything_else(value):
    """MAP-Elites ist der Bezug und im Live-Lauf nicht wählbar."""
    with pytest.raises(ValueError):
        P.SETTING_SPECS["method_select"].caster(value)


def test_method_tables_are_consistent():
    assert set(C.METHOD_LABELS) == set(C.METHOD_GROUP) == set(C.METHODS) and set(C.CMA_METHODS) | {"me"} == set(C.METHODS) and set(C.ALPHAS) < set(C.CMA_METHODS)
    assert set(C.METHOD_GROUP.values()) == set(C.GROUP_LABELS) and C.POP % C.N_EMITTERS == 0 and set(C.SIGMA_METHODS) <= set(C.CMA_METHODS)


def test_case_tables_are_consistent():
    assert set(C.CASES) == set(C.DIM_CASES) | set(C.MAZE_CASES) and set(C.CASE_LABELS) == set(C.CASES) and [C.DIM_CASES[f"d{s}"] for s in C.STEP_OPTIONS] == [(0, s) for s in C.STEP_OPTIONS]
    assert C.SIGMA_CASES == {"s40": (0, 40), "s140": (0, 140)} and C.DEFAULT_SIGMA in C.SIGMAS and C.DEFAULT_STEPS in C.STEP_OPTIONS


def test_formatters():
    assert C.fmt_int(40000) == "40 000" and C.fmt_med(None) == "–" and C.fmt_med(988.5) == "988.5" and C.fmt_med(1195.0) == "1 195"

"""Suche: Reproduzierbarkeit je Verfahren, Zählungen, Kurven, gespeichertes Archiv, Konditionszahlen, Neustarts, wer das Ziel sieht, Genomlänge unabhängig von der globalen Konstanten."""

import numpy as np
import pytest

import cme_constants as C
import cme_maze as M
import cme_search as S


def _run(method, seed=3, level=0, steps=10, gens=30, **kw):
    return S.run_search(M.make_maze(level), seed, method, steps=steps, gens=gens, **kw)


def test_unknown_method_is_refused_and_elite_tracks_need_a_stored_archive():
    with pytest.raises(ValueError):
        _run("zufall", gens=2)
    with pytest.raises(ValueError):
        S.elite_tracks(M.make_maze(0), _run("me", gens=2))


@pytest.mark.parametrize("method", C.METHODS)
def test_runs_are_reproducible_and_depend_on_the_seed(method):
    a, b, c = _run(method, 4, gens=8), _run(method, 4, gens=8), _run(method, 5, gens=8)
    assert a.coverage == b.coverage and a.qd_score == b.qd_score and a.first_solved == b.first_solved and a.restarts == b.restarts and a.qd_score != c.qd_score


@pytest.mark.parametrize("method", C.METHODS)
def test_counts_and_curves_have_the_right_shape_and_never_get_worse(method):
    res = _run(method, 3, level=3, steps=20, gens=26, keep_archive=True)
    assert res.evaluations == 26 * C.POP and len(res.coverage) == len(res.qd_score) == len(res.best_quality) == len(res.best_distance) == len(res.best_progress) == 26 and res.total == 144
    assert all(y >= x for x, y in zip(res.coverage, res.coverage[1:])) and all(y >= x - 1e-12 for x, y in zip(res.qd_score, res.qd_score[1:])) and all(y >= x for x, y in zip(res.best_quality, res.best_quality[1:]))
    assert all(y <= x for x, y in zip(res.best_progress, res.best_progress[1:])) and res.archive_q.shape == (12, 12) and res.archive_g.shape == (12, 12, 20, 2) and res.coverage[-1] <= 144


def test_condition_numbers_are_recorded_for_the_cma_methods_at_the_milestones_reached():
    for method in C.CMA_METHODS:
        res = _run(method, gens=30)
        assert sorted(res.conditions) == [10, 25] and all(v >= 0 for v in res.conditions.values())
    assert _run("me", gens=30).conditions == {} and _run("me", gens=30).restarts == 0


@pytest.mark.parametrize("method", C.METHODS)
def test_the_stored_archive_is_consistent_with_its_rides(method):
    """Qualität und Zelle jedes Elites stimmen mit einer frischen Fahrt der gespeicherten Steuerfolge überein; belegte Zellen = Abdeckung; Summe = QD-Score."""
    grid = M.make_maze(3)
    res = S.run_search(grid, 2, method, steps=15, gens=12, keep_archive=True)
    q, tracks, cells = S.elite_tracks(grid, res)
    assert len(q) == res.coverage[-1] and S.directness(tracks) == pytest.approx(q, abs=1e-12) and float(q.sum()) == pytest.approx(res.qd_score[-1])
    ci, cj = S.cell_indices(tracks[:, -1, :])
    assert list(ci) == list(cells[:, 0]) and list(cj) == list(cells[:, 1]) and tracks.shape[1] == 16


def test_all_methods_share_the_initial_population_so_the_first_generation_is_identical():
    first = {m: _run(m, 7, level=3, steps=30, gens=1) for m in C.METHODS}
    assert len({(r.coverage[0], r.qd_score[0], r.best_quality[0]) for r in first.values()}) == 1


def test_no_method_knows_the_goal_so_the_archive_curves_do_not_react_to_a_moved_goal():
    base = {m: _run(m, 2, level=3, steps=20, gens=12) for m in C.METHODS}
    old = C.GOAL
    try:
        C.GOAL = (20.5, 20.5)
        moved = {m: _run(m, 2, level=3, steps=20, gens=12) for m in C.METHODS}
    finally:
        C.GOAL = old
    for m in C.METHODS:
        assert base[m].coverage == moved[m].coverage and base[m].qd_score == moved[m].qd_score, m


def test_a_run_with_another_genome_length_leaves_the_global_constant_alone_and_the_dimension_follows_the_length():
    res = _run("mae10", steps=33, gens=5, keep_archive=True)
    assert C.STEPS == 140 and res.archive_g.shape[2] == 33


def test_the_methods_with_different_learning_rates_run_differently_and_alpha_is_the_documented_value():
    runs = {m: _run(m, 1, steps=20, gens=40).qd_score for m in ("mae01", "mae10", "mae50")}
    assert runs["mae01"] != runs["mae10"] != runs["mae50"] and C.ALPHAS == {"mae01": 0.01, "mae10": 0.1, "mae50": 0.5}


def test_the_sigma_matters_for_the_cma_methods_but_not_for_map_elites():
    grid = M.make_maze(0)
    a, b = (S.run_search(grid, 1, "cmame", steps=10, sigma0=s, gens=15).qd_score for s in (0.05, 0.4))
    m1, m2 = (S.run_search(grid, 1, "me", steps=10, sigma0=s, gens=15).qd_score for s in (0.05, 0.4))
    assert a != b and m1 == m2


def test_stop_when_solved_ends_the_run_early():
    res = S.run_search(M.make_maze(0), 3, "me", steps=140, gens=300, stop_when_solved=True)
    assert res.first_solved is not None and len(res.coverage) < 300 and len(res.coverage) == (res.first_solved - 1) // C.POP + 1


def test_map_elites_equals_the_fifth_piece():
    """MAP-Elites (Zellgröße 2) auf Stufe 3, 140 Schritte, 400 Generationen, Seeds 0 bis 3: gleiche Evaluationen bis zum Erfolg und gleicher QD-Score wie in Stück 5 (map-elites-demo)."""
    grid = M.make_maze(3)
    res = [S.run_search(grid, s, "me") for s in range(4)]
    assert [r.first_solved for r in res] == [863, 2383, 2035, 1128] and [round(r.qd_score[-1], 6) for r in res] == [102.467452, 103.59059, 103.420297, 102.867503]

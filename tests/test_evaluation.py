"""Auswertung: Täuschungsmaße je Stufe, Lösbarkeit, Live-Paar, Hilfsfunktionen der Studie, Vollständigkeit und Stimmigkeit der vorgerechneten Datei (inklusive Neurechnung einzelner Läufe)."""

import numpy as np
import pytest
from scipy.stats import mannwhitneyu

import cme_constants as C
import cme_evaluation as E
import cme_maze as M
import cme_search as S

PRE = E.load_precomputed()


def test_level_metrics_by_hand():
    m0, m3 = E.level_metrics(0), E.level_metrics(3)
    assert m0["path"] == 17 and m0["straight"] == pytest.approx(17.0) and m0["detour"] == pytest.approx(1.0) and m0["minima"] == 0 and m0["trap"] is None and m0["free"] == 22 * 22
    assert m3["path"] == 49 and m3["detour"] == pytest.approx(49 / 17) and m3["minima"] == 1 and m3["trap"] == pytest.approx(9.0) and m3["free"] == 22 * 22 - 19


@pytest.mark.parametrize("level", list(C.LEVELS))
def test_every_level_is_solvable_within_the_step_budget(level):
    dist, steps = E.solvability(level)
    assert dist < 0.5 and steps <= C.STEPS


def test_live_pair_runs_map_elites_and_the_chosen_method_from_the_same_start():
    (me, q1, t1, c1), (cm, q2, t2, c2) = E.live_pair(0, 12, 5, 20, 0.2, "mae10")
    assert me.total == cm.total == 144 and len(q1) == me.coverage[-1] and len(q2) == cm.coverage[-1] and t1.shape == (len(q1), 21, 2) and t2.shape == (len(q2), 21, 2)
    assert me.coverage[0] == cm.coverage[0] and me.qd_score[0] == cm.qd_score[0]


def test_success_helpers_by_hand():
    runs = [100, None, 500, 50, None]
    assert E.success_count(runs) == 3 and E.success_count(runs, 100) == 2 and E.median_solved(runs) == 100.0 and E.median_solved([None, None]) is None


def test_mann_whitney_agrees_with_scipy():
    x, y = [1.0, 2.0, 3.0, 4.0, 5.0], [3.5, 4.5, 5.5, 6.5, 7.5]
    assert E.mannwhitney_p(x, y) == pytest.approx(mannwhitneyu(x, y, alternative="two-sided").pvalue) and E.mannwhitney_p(x, x) == pytest.approx(1.0)


def test_the_study_covers_every_case_and_every_method():
    assert set(PRE["cases"]) == set(C.CASES) and PRE["budget"] == C.STUDY_BUDGET
    for case, (level, steps) in C.CASES.items():
        info = E.case_info(PRE, case)
        assert (info["level"], info["steps"], info["seeds"], info["total"]) == (level, steps, C.STUDY_SEEDS, 144) and set(info["methods"]) == set(C.METHODS)
        for m in C.METHODS:
            c = E.study_cell(PRE, case, m)
            n = C.STUDY_SEEDS
            assert len(c["runs"]) == len(c["progress"]) == len(c["best_quality"]) == len(c["coverage"]) == len(c["qd"]) == len(c["restarts"]) == n
            assert len(c["mean_coverage"]) == len(c["mean_qd"]) == C.STUDY_GENS and max(c["coverage"]) <= 144 and min(c["qd"]) >= 0 and all(0.0 <= q <= 1.0 for q in c["best_quality"])
            assert (c["log_condition"] == {}) == (m == "me") and (m == "me" or set(c["log_condition"]) == {str(g) for g in C.MILESTONES})


def test_the_sigma_study_covers_every_cell():
    assert set(PRE["sigma"]) == set(C.SIGMA_CASES)
    for case in C.SIGMA_CASES:
        assert set(PRE["sigma"][case]) == set(C.SIGMA_METHODS)
        for m in C.SIGMA_METHODS:
            assert set(PRE["sigma"][case][m]) == {str(s) for s in C.SIGMAS}
            for sg in C.SIGMAS:
                c = E.sigma_cell(PRE, case, m, sg)
                assert len(c["coverage"]) == len(c["qd"]) == C.STUDY_SEEDS


def test_the_default_sigma_in_the_sigma_study_equals_the_case_study():
    """σ₀ = 0.2 ist die Standardeinstellung: dieselben Läufe in der Fall-Studie (freies Feld mit 40 und 140 Schritten) und im Sweep."""
    for case, steps in (("s40", 40), ("s140", 140)):
        for m in C.SIGMA_METHODS:
            assert E.sigma_cell(PRE, case, m, C.DEFAULT_SIGMA)["qd"] == E.study_cell(PRE, f"d{steps}", m)["qd"]


def test_the_stored_level_metrics_equal_the_exact_ones():
    for case in C.CASES:
        stored, fresh = E.case_info(PRE, case)["metrics"], E.level_metrics(C.CASES[case][0])
        for key in ("path", "straight", "detour", "fdc", "minima", "free"):
            assert stored[key] == pytest.approx(fresh[key])


def within(value, values, k=4.0, floor=1.0):
    """Liegt `value` höchstens k Standardabweichungen (mindestens `floor`) vom Mittel der gespeicherten Werte?"""
    values = np.asarray(values, float)
    return abs(value - values.mean()) <= k * max(float(values.std(ddof=1)), floor)


def test_a_map_elites_study_run_is_reproduced_exactly_by_a_fresh_run_with_the_same_seed():
    """MAP-Elites braucht keine Eigenzerlegung und ist auf jeder Plattform bitgleich: alle gespeicherten Werte eines Laufs stimmen exakt mit einem frischen Lauf überein."""
    for case, seed in (("d20", 0), ("d40", 3), ("l3", 5)):
        level, steps = C.CASES[case]
        res = S.run_search(M.make_maze(level), seed, "me", steps=steps, sigma0=C.DEFAULT_SIGMA, gens=C.STUDY_GENS)
        c = E.study_cell(PRE, case, "me")
        assert res.first_solved == c["runs"][seed] and res.best_progress[-1] == c["progress"][seed] and res.best_quality[-1] == pytest.approx(c["best_quality"][seed], abs=1e-6)
        assert res.coverage[-1] == c["coverage"][seed] and res.qd_score[-1] == pytest.approx(c["qd"][seed], abs=1e-5) and res.restarts == c["restarts"][seed] == 0


@pytest.mark.parametrize("case,method,seed", [("d20", "cmame", 1), ("d20", "mae10", 2), ("d40", "mae01", 3), ("d10", "mae50", 4), ("l3", "cmame", 5)])
def test_a_cma_study_run_falls_into_the_stored_distribution_when_run_again(case, method, seed):
    """CMA-Läufe sind nicht plattformübergreifend bitgleich: die Eigenzerlegung (LAPACK) liefert je Plattform andere Eigenvektoren (Vorzeichen, Drehung bei fast gleichen Eigenwerten), und ein Emitter, dessen Kovarianz nahe der Einheitsmatrix bleibt, zieht dann
    andere Stichproben. Ein frischer Lauf mit demselben Seed ist deshalb ein anderer Zug aus derselben Verteilung: QD-Score, Abdeckung und Neustarts liegen innerhalb von 4 Standardabweichungen der 20 gespeicherten Läufe."""
    level, steps = C.CASES[case]
    res = S.run_search(M.make_maze(level), seed, method, steps=steps, sigma0=C.DEFAULT_SIGMA, gens=C.STUDY_GENS)
    c = E.study_cell(PRE, case, method)
    assert within(res.qd_score[-1], c["qd"]) and within(res.coverage[-1], c["coverage"], floor=2.0) and within(res.restarts, c["restarts"], floor=5.0) and 0.0 <= res.best_quality[-1] <= 1.0 + 1e-9               # Direktheit kann durch Rundung um 1e-16 über 1 liegen


@pytest.mark.parametrize("case,method,sigma", [("s40", "cmame", 0.05), ("s140", "mae10", 0.4)])
def test_a_sigma_run_falls_into_the_stored_distribution_when_run_again(case, method, sigma):
    steps = C.SIGMA_CASES[case][1]
    res = S.run_search(M.make_maze(0), 2, method, steps=steps, sigma0=sigma, gens=C.STUDY_GENS)
    c = E.sigma_cell(PRE, case, method, sigma)
    assert within(res.qd_score[-1], c["qd"], floor=2.0) and within(res.coverage[-1], c["coverage"], floor=3.0)


def test_the_mean_curves_are_the_mean_over_the_runs_and_never_get_worse():
    c = E.study_cell(PRE, "d40", "mae10")
    assert all(y >= x for x, y in zip(c["mean_coverage"], c["mean_coverage"][1:])) and all(y >= x - 1e-9 for x, y in zip(c["mean_qd"], c["mean_qd"][1:]))
    assert c["mean_coverage"][-1] == pytest.approx(np.mean(c["coverage"]), abs=1e-3) and c["mean_qd"][-1] == pytest.approx(np.mean(c["qd"]), abs=1e-3)

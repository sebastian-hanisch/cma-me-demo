"""JEDE Zahl aus README und App-Texten wird hier nachgerechnet: Täuschungsmaße exakt aus dem Labyrinth, Studienzahlen aus der vorgerechneten Datei (je Zelle 20 Läufe, Seeds 0 bis 19, 40 000 Evaluationen, Anfangsschrittweite 0.2, Zellgröße 2).
Die Datei ist fest; ändern sich die Zahlen nach einer neuen Rechnung, müssen README und Hilfetexte nachgezogen werden."""

import numpy as np
import pytest

import cme_constants as C
import cme_evaluation as E

PRE = E.load_precomputed()
CASES = list(C.CASES)


def cell(case, m):
    return E.study_cell(PRE, case, m)


def qd(case, m):
    return np.array(cell(case, m)["qd"], float)


def gain(case, m):
    return 100 * (qd(case, m).mean() / qd(case, "me").mean() - 1)


def test_exact_deception_measures_and_archive_size():
    """README: Stufe 3 / 4: kürzester Weg 49 / 77 Zellen, Umwegfaktor 2.88 / 4.53, Korrelation 0.69 / 0.68, Fallen 1 / 2; je 144 Archivzellen mit freier Fläche."""
    m3, m4 = E.level_metrics(3), E.level_metrics(4)
    assert (m3["path"], m4["path"]) == (49, 77) and (round(m3["detour"], 2), round(m4["detour"], 2)) == (2.88, 4.53) and (round(m3["fdc"], 2), round(m4["fdc"], 2)) == (0.69, 0.68) and (m3["minima"], m4["minima"]) == (1, 2)
    assert all(PRE["cases"][c]["total"] == 144 for c in CASES)


def test_qd_score_of_map_elites_and_the_four_cma_methods_in_every_case():
    """README: QD-Score (Mittel ± Standardabweichung) MAP-Elites: 20.30 (10 Schritte) / 51.90 / 127.34 / 133.91 / 121.76 (140 Schritte), Stufe 3 102.96, Stufe 4 59.29; CMA-MAE 0.1: 28.92 / 70.67 / 138.95 / 138.21 / 121.18, 100.75, 57.58;
    CMA-ME: 23.28 / 58.21 / 128.69 / 132.78 / 120.20, 101.08, 58.44; CMA-MAE 0.01: 30.75 / 71.84 / 138.78 / 136.22 / 117.86, 97.88, 56.65; CMA-MAE 0.5: 25.27 / 66.72 / 133.76 / 136.77 / 120.82, 100.51, 56.51."""
    expect = {"me": [20.30, 51.90, 127.34, 133.91, 121.76, 102.96, 59.29], "mae10": [28.92, 70.67, 138.95, 138.21, 121.18, 100.75, 57.58], "cmame": [23.28, 58.21, 128.69, 132.78, 120.20, 101.08, 58.44],
              "mae01": [30.75, 71.84, 138.78, 136.22, 117.86, 97.88, 56.65], "mae50": [25.27, 66.72, 133.76, 136.77, 120.82, 100.51, 56.51]}
    assert CASES == ["d10", "d20", "d40", "d80", "d140", "l3", "l4"]
    for m, values in expect.items():
        assert [round(float(qd(c, m).mean()), 2) for c in CASES] == values, m


def test_standard_deviations_of_map_elites_and_cma_mae_on_few_dimensions():
    """README: Standardabweichung des QD-Scores bei 10 / 20 Schritten: MAP-Elites 2.66 / 5.09, CMA-MAE 0.1 2.40 / 1.59, CMA-MAE 0.01 1.02 / 0.65."""
    assert [round(float(qd(c, "me").std(ddof=1)), 2) for c in ("d10", "d20")] == [2.66, 5.09]
    assert [round(float(qd(c, "mae10").std(ddof=1)), 2) for c in ("d10", "d20")] == [2.40, 1.59] and [round(float(qd(c, "mae01").std(ddof=1)), 2) for c in ("d10", "d20")] == [1.02, 0.65]


def test_the_gain_over_map_elites_shrinks_with_the_dimension():
    """README: Gewinn des QD-Scores gegenüber MAP-Elites bei 20 / 40 / 80 / 160 / 280 Dimensionen: CMA-ME +14.7 / +12.2 / +1.1 / −0.8 / −1.3 %, CMA-MAE 0.01 +51.5 / +38.4 / +9.0 / +1.7 / −3.2 %,
    0.1 +42.5 / +36.2 / +9.1 / +3.2 / −0.5 %, 0.5 +24.5 / +28.6 / +5.0 / +2.1 / −0.8 %; Stufe 3 (CMA-ME / 0.01 / 0.1 / 0.5): −1.8 / −4.9 / −2.2 / −2.4 %, Stufe 4: −1.4 / −4.4 / −2.9 / −4.7 %."""
    expect = {"cmame": [14.7, 12.2, 1.1, -0.8, -1.3, -1.8, -1.4], "mae01": [51.5, 38.4, 9.0, 1.7, -3.2, -4.9, -4.4], "mae10": [42.5, 36.2, 9.1, 3.2, -0.5, -2.2, -2.9], "mae50": [24.5, 28.6, 5.0, 2.1, -0.8, -2.4, -4.7]}
    for m, values in expect.items():
        assert [round(gain(c, m), 1) for c in CASES] == pytest.approx(values, abs=0.051), m
    for m in C.CMA_METHODS:                                                             # ab 40 Dimensionen sinkt der Gewinn mit jeder Verdopplung (bei 20 Dimensionen ist er bei CMA-MAE 0.5 noch kleiner als bei 40)
        gains = [gain(c, m) for c in ("d20", "d40", "d80", "d140")]
        assert gains == sorted(gains, reverse=True)


def test_the_significance_of_the_gain_and_of_the_losses():
    """README: Mann-Whitney gegen MAP-Elites: alle Gewinne bei 20 und 40 Dimensionen p < 0.001; CMA-ME bei 80 Dimensionen +1.1 % ist nicht belegt (p = 0.19); bei 280 Dimensionen sind alle vier Verfahren belegt schlechter (p < 0.001), auf Stufe 3 ebenso,
    auf Stufe 4 CMA-ME mit p = 0.047 ein Grenzfall."""
    p = lambda c, m: E.mannwhitney_p(qd(c, m), qd(c, "me"))
    assert all(p(c, m) < 0.001 for c in ("d10", "d20") for m in C.CMA_METHODS)
    assert round(p("d40", "cmame"), 2) == 0.19 and all(p("d40", m) < 0.001 for m in ("mae01", "mae10", "mae50"))
    assert all(p("d140", m) < 0.001 for m in C.CMA_METHODS) and all(p("l3", m) < 0.001 for m in C.CMA_METHODS)
    assert round(p("l4", "cmame"), 3) == 0.047 and all(p("l4", m) < 0.001 for m in ("mae01", "mae10", "mae50"))


def test_on_few_dimensions_the_gain_comes_mostly_from_more_cells():
    """README: Abdeckung bei 20 Schritten (von 144): MAP-Elites 52.15, CMA-ME 58.75, CMA-MAE 0.01 / 0.1 / 0.5: 72.25 / 70.90 / 67.00; ab 80 Schritten belegen alle Verfahren (fast) alle 144 Zellen."""
    assert [round(float(np.mean(cell("d20", m)["coverage"])), 2) for m in C.METHODS] == [52.15, 58.75, 72.25, 70.90, 67.00]
    assert all(np.mean(cell("d140", m)["coverage"]) == 144.0 for m in C.METHODS) and round(float(np.mean(cell("d80", "mae10")["coverage"])), 2) == 143.95


def test_the_maze_at_native_dimension():
    """README: Stufe 3: alle Verfahren lösen 20 von 20 Läufen; belegte Zellen MAP-Elites 144, CMA-MAE 0.01 143.5; Stufe 4 bei 140 Schritten: kein Verfahren löst einen Lauf, belegte Zellen MAP-Elites 85.9, CMA-ME 85.7, CMA-MAE 0.01 / 0.1 / 0.5: 82.8 / 82.5 / 79.9,
    Restweg MAP-Elites 29.75, CMA-ME 29.95, CMA-MAE 32.45 / 32.55 / 33.75."""
    assert all(E.success_count(cell("l3", m)["runs"]) == 20 for m in C.METHODS) and all(E.success_count(cell("l4", m)["runs"]) == 0 for m in C.METHODS)
    assert [round(float(np.mean(cell("l3", m)["coverage"])), 2) for m in ("me", "mae01")] == [144.0, 143.5]
    assert [round(float(np.mean(cell("l4", m)["coverage"])), 2) for m in C.METHODS] == [85.9, 85.7, 82.8, 82.5, 79.9]
    assert [round(float(np.mean(cell("l4", m)["progress"])), 2) for m in C.METHODS] == [29.75, 29.95, 32.45, 32.55, 33.75]


def test_cma_me_restarts_almost_all_the_time_and_cma_mae_rarely():
    """README: CMA-ME: im Mittel 1 323 bis 1 809 Neustarts je Lauf (von 1 995 Emitter-Generationen: 5 Emitter · 399 Generationen), CMA-MAE 41 bis 53; MAP-Elites kennt keine Neustarts."""
    cm = [float(np.mean(cell(c, "cmame")["restarts"])) for c in CASES]
    ma = [float(np.mean(cell(c, m)["restarts"])) for c in CASES for m in ("mae01", "mae10", "mae50")]
    assert (round(min(cm)), round(max(cm))) == (1323, 1809) and (round(min(ma)), round(max(ma))) == (41, 53) and C.N_EMITTERS * (C.STUDY_GENS - 1) == 1995
    assert all(r == 0 for c in CASES for r in cell(c, "me")["restarts"])


def test_the_condition_number_of_the_covariance_stays_small_and_shrinks_with_the_dimension():
    """README: Zehnerlogarithmus der Konditionszahl nach 400 Generationen: CMA-ME 0.00 in jedem Fall (kugelförmig, nichts gelernt); CMA-MAE 0.1 bei 20 / 40 / 80 / 160 / 280 Dimensionen: 0.44 / 0.21 / 0.10 / 0.06 / 0.02, also Konditionszahlen von 2.8 bis 1.05."""
    assert all(cell(c, "cmame")["log_condition"]["400"] < 0.005 for c in CASES)
    got = [cell(c, "mae10")["log_condition"]["400"] for c in ("d10", "d20", "d40", "d80", "d140")]
    assert [round(x, 2) for x in got] == [0.44, 0.21, 0.10, 0.06, 0.02] and round(10 ** got[0], 1) == 2.8 and round(10 ** got[-1], 2) == 1.05


def test_sigma_sweep():
    """README: QD-Score nach 400 Generationen bei σ₀ = 0.05 / 0.1 / 0.2 / 0.4: freies Feld mit 40 Schritten CMA-ME 97.92 / 127.28 / 128.69 / 123.28, CMA-MAE 0.1 126.65 / 136.00 / 138.95 / 136.88; mit 140 Schritten CMA-ME 117.71 / 121.96 / 120.20 / 114.28,
    CMA-MAE 0.1 108.84 / 120.58 / 121.18 / 117.06."""
    got = {(case, m): [round(float(np.mean(E.sigma_cell(PRE, case, m, sg)["qd"])), 2) for sg in C.SIGMAS] for case in ("s40", "s140") for m in C.SIGMA_METHODS}
    assert got == {("s40", "cmame"): [97.92, 127.28, 128.69, 123.28], ("s40", "mae10"): [126.65, 136.0, 138.95, 136.88], ("s140", "cmame"): [117.71, 121.96, 120.2, 114.28], ("s140", "mae10"): [108.84, 120.58, 121.18, 117.06]}


def test_with_a_small_enough_sigma_cma_me_at_280_dimensions_catches_up_with_map_elites_but_does_not_pass_it():
    """README: bei 140 Schritten erreicht CMA-ME mit der besten Schrittweite (0.1) 121.96 gegen 121.76 bei MAP-Elites (+0.2 %); das ist kein belegter Vorsprung."""
    best = max(float(np.mean(E.sigma_cell(PRE, "s140", "cmame", sg)["qd"])) for sg in C.SIGMAS)
    me = float(qd("d140", "me").mean())
    assert round(best, 2) == 121.96 and round(100 * (best / me - 1), 1) == 0.2
    assert E.mannwhitney_p(E.sigma_cell(PRE, "s140", "cmame", 0.1)["qd"], qd("d140", "me")) > 0.05


def test_map_elites_equals_the_fifth_piece():
    """README: MAP-Elites auf Stufe 3 (102.96) und Stufe 4 (59.29) hat dieselben Mittelwerte wie in Stück 5; Abdeckung 144.00 und 85.90."""
    assert round(float(qd("l3", "me").mean()), 2) == 102.96 and round(float(qd("l4", "me").mean()), 2) == 59.29
    assert float(np.mean(cell("l3", "me")["coverage"])) == 144.0 and round(float(np.mean(cell("l4", "me")["coverage"])), 2) == 85.9


def test_preset_help_quotes_the_study():
    """PRESET_HELP: 20 Schritte (QD 70.67 gegen 51.90 = +36 %, p < 0.001; Zellen 70.90 gegen 52.15), 140 Schritte (121.18 gegen 121.76 = −0.5 %), Falle (CMA-ME 101.08 gegen 102.96 = −1.8 %, beide 144 Zellen, 20 von 20),
    σ₀ = 0.05 (97.92 gegen 128.69 bei 0.2 und 127.34 bei MAP-Elites)."""
    assert round(float(qd("d20", "mae10").mean()), 2) == 70.67 and round(float(qd("d20", "me").mean()), 2) == 51.90 and round(gain("d20", "mae10")) == 36
    assert round(float(np.mean(cell("d20", "mae10")["coverage"])), 2) == 70.90 and round(float(np.mean(cell("d20", "me")["coverage"])), 2) == 52.15
    assert round(float(qd("d140", "mae10").mean()), 2) == 121.18 and round(gain("d140", "mae10"), 1) == -0.5
    assert round(float(qd("l3", "cmame").mean()), 2) == 101.08 and round(gain("l3", "cmame"), 1) == -1.8 and float(np.mean(cell("l3", "cmame")["coverage"])) == 144.0
    assert E.success_count(cell("l3", "cmame")["runs"]) == 20 == E.success_count(cell("l3", "me")["runs"])
    assert round(float(np.mean(E.sigma_cell(PRE, "s40", "cmame", 0.05)["qd"])), 2) == 97.92 and round(float(qd("d40", "cmame").mean()), 2) == 128.69 and round(float(qd("d40", "me").mean()), 2) == 127.34
    assert C.STUDY_BUDGET == 40_000
    for name, text in C.PRESET_HELP.items():
        assert text and name in C.PRESETS

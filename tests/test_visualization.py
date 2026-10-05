"""Abbildungen: gesperrte Achsen, Spuren, Beschriftungen, Farben, Fehlerbalken."""

import numpy as np
import pytest

import cme_constants as C
import cme_evaluation as E
import cme_maze as M
import cme_visualization as V

PRE = E.load_precomputed()


def _locked(fig):
    return all(ax.fixedrange for ax in fig.select_xaxes()) and all(ax.fixedrange for ax in fig.select_yaxes())


def test_all_charts_lock_their_axes():
    grid = M.make_maze(0)
    (me, q1, t1, _), (cm, q2, t2, _) = E.live_pair(0, 10, 1, 20, 0.2, "cmame")
    figs = [V.build_archive_map(grid, me.archive_q), V.build_archive_map(grid, cm.archive_q), V.build_paths_figure(grid, t1, q1), V.build_paths_figure(grid, t2, q2),
            V.build_coverage_curve(me.coverage, cm.coverage, "cmame", me.total), V.build_qd_curve(me.qd_score, cm.qd_score, "mae50"), V.build_cases_chart(PRE), V.build_sigma_chart(PRE),
            *[V.build_study_curve(PRE, case) for case in C.CASES], *[V.build_condition_chart(PRE, case) for case in C.CASES]]
    assert all(_locked(f) for f in figs)


def test_archive_map_shows_the_quality_of_filled_cells_and_leaves_empty_cells_out():
    grid = M.make_maze(3)
    (me, _, _, _), _ = E.live_pair(3, 10, 1, 40, 0.2, "mae10")
    fig = V.build_archive_map(grid, me.archive_q)
    heat = fig.data[0]
    z = np.array(heat.z).T
    assert heat.dx == C.CELL and heat.x0 == C.CELL / 2 and np.isnan(z[me.archive_q < 0]).all() and (z[me.archive_q >= 0] == me.archive_q[me.archive_q >= 0]).all()
    walls = np.array(fig.data[1].z).T
    assert (walls[grid] == 1.0).all() and np.isnan(walls[~grid]).all()


def test_paths_figure_has_one_line_per_elite_and_one_end_marker_each():
    grid = M.make_maze(0)
    _, (cm, q, tracks, _) = E.live_pair(0, 10, 1, 20, 0.2, "mae10")
    fig = V.build_paths_figure(grid, tracks, q)
    lines = next(t for t in fig.data if t.name == "Fahrten der Elites")
    ends = next(t for t in fig.data if t.name == "Endpunkte der Elites")
    assert len(ends.x) == len(q) == cm.coverage[-1] and sum(1 for v in lines.x if v is None) == len(q) and list(ends.marker.color) == list(q)


def test_pair_curves_name_both_methods_and_the_colors_are_fixed():
    fig = V.build_qd_curve([1.0, 2.0], [0.5, 1.0], "mae10")
    assert [t.name for t in fig.data] == [C.METHOD_LABELS["me"], C.METHOD_LABELS["mae10"]] and fig.data[0].line.color == V.METHOD_COLORS["me"] and fig.data[1].line.color == V.METHOD_COLORS["mae10"]
    cov = V.build_coverage_curve([10, 20], [5, 10], "cmame", 40)
    assert list(cov.data[0].y) == [0.25, 0.5] and list(cov.data[1].y) == [0.125, 0.25]


def test_cases_chart_has_one_series_per_method_over_all_cases_and_error_bars_equal_the_standard_deviation():
    fig = V.build_cases_chart(PRE)
    assert [t.name for t in fig.data] == [C.METHOD_LABELS[m] for m in C.METHODS] and all(len(t.x) == len(C.CASES) for t in fig.data) and list(fig.data[0].x) == [C.CASE_LABELS[c] for c in C.CASES]
    vals = np.array(E.study_cell(PRE, "d40", "mae10")["qd"])
    i = list(C.CASES).index("d40")
    mae = fig.data[C.METHODS.index("mae10")]
    assert mae.y[i] == pytest.approx(vals.mean()) and mae.error_y.array[i] == pytest.approx(vals.std(ddof=1))


def test_study_curve_has_one_line_per_method_over_the_evaluations():
    fig = V.build_study_curve(PRE, "d40")
    assert [t.name for t in fig.data] == [C.METHOD_LABELS[m] for m in C.METHODS] and fig.data[0].x[0] == C.POP and fig.data[0].x[-1] == C.STUDY_BUDGET


def test_condition_chart_has_one_line_per_cma_method_over_the_milestones():
    fig = V.build_condition_chart(PRE, "d20")
    assert [t.name for t in fig.data] == [C.METHOD_LABELS[m] for m in C.CMA_METHODS] and all(list(t.x) == list(C.MILESTONES) for t in fig.data)


def test_sigma_chart_has_a_line_for_each_method_and_length_and_a_reference_line_for_map_elites():
    fig = V.build_sigma_chart(PRE)
    assert len(fig.data) == len(C.SIGMA_METHODS) * len(C.SIGMA_CASES) and all(len(t.x) == len(C.SIGMAS) for t in fig.data)
    assert len(fig.layout.shapes) == 2 and {round(float(s.y0), 6) for s in fig.layout.shapes} == {round(float(np.mean(E.study_cell(PRE, f"d{st}", "me")["qd"])), 6) for st in (40, 140)}

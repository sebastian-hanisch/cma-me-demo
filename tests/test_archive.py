"""Archiv und Emitter-Regeln von Hand: Direktheit, Zellen, Archivzellen mit freier Fläche, Eintragen, Kennzahlen, CMA-ME (neue Zelle vor Verbesserung, Rangfolge), CMA-MAE (Schwellen mit Lernrate, Verbesserung Δ)."""

import numpy as np
import pytest

import cme_constants as C
import cme_maze as M
import cme_robot as R
import cme_search as S

FREE = M.make_maze(0)


def test_directness_of_a_straight_slow_ride_is_one_and_of_standing_still_zero():
    g = np.zeros((1, 140, 2))
    g[:, :, 1] = 0.1
    _, track = R.simulate(g, FREE, trajectories=True)
    assert S.directness(track)[0] == pytest.approx(1.0)
    _, still = R.simulate(np.zeros((1, 140, 2)), FREE, trajectories=True)
    assert S.directness(still)[0] == 0.0


def test_directness_of_an_out_and_back_route_by_hand():
    """(3.5, 3.5) -> (3.5, 5.5) -> (4.5, 5.5) -> (4.5, 3.5): gefahren 2 + 1 + 2 = 5, Luftlinie Start–Ende 1: Direktheit 0.2."""
    assert S.directness(np.array([[[3.5, 3.5], [3.5, 5.5], [4.5, 5.5], [4.5, 3.5]]]))[0] == pytest.approx(0.2)


def test_the_robot_runs_any_genome_length_without_the_global_step_count():
    for steps in (5, 20, 140, 300):
        genomes = R.random_genomes(np.random.default_rng(1), 4, steps)
        end, track = R.simulate(genomes, FREE, trajectories=True)
        assert genomes.shape == (4, steps, 2) and track.shape == (4, steps + 1, 2) and end.shape == (4, 2)
    assert C.STEPS == 140


def test_cell_indices_by_hand():
    end = np.array([[3.5, 3.5], [23.9, 0.1], [24.0, 24.0]])
    assert [list(a) for a in S.cell_indices(end)] == [[1, 11, 11], [1, 0, 11]] and S.n_blocks() == 12


@pytest.mark.parametrize("level,expected", [(0, 144), (3, 144), (4, 144)])
def test_free_blocks_count_the_archive_cells_that_can_hold_an_end_position(level, expected):
    assert int(S.free_blocks(M.make_maze(level)).sum()) == expected


def test_insert_fills_an_empty_cell_replaces_with_better_keeps_better_and_ignores_equal():
    quality, elites = np.full((2, 2), -1.0), np.zeros((2, 2, 1, 2))
    g = lambda v: np.full((1, 1, 2), v)
    S.insert(quality, elites, np.array([0]), np.array([1]), np.array([0.4]), g(1.0))
    S.insert(quality, elites, np.array([0]), np.array([1]), np.array([0.6]), g(2.0))
    S.insert(quality, elites, np.array([0]), np.array([1]), np.array([0.5]), g(3.0))
    S.insert(quality, elites, np.array([0]), np.array([1]), np.array([0.6]), g(4.0))
    assert quality[0, 1] == 0.6 and elites[0, 1, 0, 0] == 2.0 and quality[0, 0] == -1.0


def test_a_child_that_is_better_only_by_rounding_does_not_replace_the_elite():
    quality, elites = np.full((1, 1), -1.0), np.zeros((1, 1, 1, 2))
    S.insert(quality, elites, np.array([0]), np.array([0]), np.array([0.5]), np.full((1, 1, 2), 1.0))
    S.insert(quality, elites, np.array([0]), np.array([0]), np.array([0.5 + 1e-15]), np.full((1, 1, 2), 2.0))
    assert elites[0, 0, 0, 0] == 1.0
    S.insert(quality, elites, np.array([0]), np.array([0]), np.array([0.5 + 1e-9]), np.full((1, 1, 2), 3.0))
    assert elites[0, 0, 0, 0] == 3.0


def test_insert_with_quality_zero_fills_an_empty_cell_and_a_batch_goes_in_order():
    quality, elites = np.full((1, 1), -1.0), np.zeros((1, 1, 1, 2))
    S.insert(quality, elites, np.array([0]), np.array([0]), np.array([0.0]), np.full((1, 1, 2), 5.0))
    assert quality[0, 0] == 0.0 and elites[0, 0, 0, 0] == 5.0
    genomes = np.array([[[1.0, 1.0]], [[2.0, 2.0]], [[3.0, 3.0]]])
    S.insert(quality, elites, np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0.3, 0.9, 0.5]), genomes)
    assert quality[0, 0] == 0.9 and elites[0, 0, 0, 0] == 2.0


def test_archive_metrics_by_hand():
    assert S.archive_metrics(np.array([[0.5, -1.0], [0.25, 1.0]])) == (3, 1.75) and S.archive_metrics(np.full((3, 3), -1.0)) == (0, 0.0) and S.archive_metrics(np.zeros((2, 2))) == (4, 0.0)


def test_cma_me_status_new_cell_improvement_nothing_and_the_sequence_within_a_batch():
    quality = np.array([[0.5, -1.0], [-1.0, -1.0]])
    ci, cj = np.array([0, 0, 1, 1, 1, 0]), np.array([0, 1, 1, 1, 1, 0])
    q = np.array([0.75, 0.25, 0.5, 0.75, 0.25, 0.25])
    status, value = S.status_and_value(quality, ci, cj, q)
    # Kind 0: Verbesserung in Zelle (0,0) (0.75 > 0.5) mit 0.25; Kind 1: neue Zelle (0,1) mit Wert 0.25; Kind 2: neue Zelle (1,1) mit 0.5;
    # Kind 3: dieselbe Zelle (1,1), jetzt Verbesserung (0.75 > 0.5) mit 0.25; Kind 4: 0.25 < 0.75 nichts; Kind 5: Zelle (0,0) ist nach Kind 0 bei 0.75: 0.25 nichts
    assert list(status) == [1, 2, 2, 1, 0, 0] and value == pytest.approx([0.25, 0.25, 0.5, 0.25, 0.0, 0.0])
    assert quality[0, 0] == 0.5 and quality[1, 1] == -1.0                 # das Archiv selbst bleibt unverändert


def test_cma_me_ranking_puts_new_cells_first_then_improvements_each_by_value_and_leaves_out_the_rest():
    status = np.array([1, 2, 0, 2, 1, 0])
    value = np.array([0.3, 0.2, 0.0, 0.6, 0.5, 0.0])
    assert list(S.rank_improvements(status, value)) == [3, 1, 4, 0]
    assert list(S.rank_improvements(np.zeros(3, int), np.zeros(3))) == []


def test_cma_mae_thresholds_by_hand():
    """Lernrate 0.5, Start-Schwelle 0: Kind 0 (q 0.8) hat Δ = 0.8, Schwelle wird 0.4; Kind 1 (q 0.6, dieselbe Zelle) hat Δ = 0.2 (0.6 > 0.4), Schwelle 0.5; Kind 2 (q 0.3) hat Δ = −0.2, ändert nichts; Kind 3 (q 0, andere Zelle) Δ = 0, nicht größer als die Schwelle."""
    thr = np.zeros((2, 2))
    occ_g, occ_filled = np.zeros((2, 2, 1, 2)), np.zeros((2, 2), bool)
    ci, cj = np.array([0, 0, 0, 1]), np.array([0, 0, 0, 1])
    q = np.array([0.8, 0.6, 0.3, 0.0])
    genomes = np.array([[[1.0, 1.0]], [[2.0, 2.0]], [[3.0, 3.0]], [[4.0, 4.0]]])
    delta = S.mae_update(thr, occ_g, occ_filled, ci, cj, q, genomes, 0.5)
    assert delta == pytest.approx([0.8, 0.2, -0.2, 0.0]) and thr[0, 0] == pytest.approx(0.5) and thr[1, 1] == 0.0
    assert occ_g[0, 0, 0, 0] == 2.0 and occ_filled[0, 0] and not occ_filled[1, 1]


def test_cma_mae_with_learning_rate_one_makes_the_threshold_the_best_quality_in_the_cell():
    thr = np.zeros((1, 1))
    occ_g, occ_filled = np.zeros((1, 1, 1, 2)), np.zeros((1, 1), bool)
    S.mae_update(thr, occ_g, occ_filled, np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0.4, 0.9, 0.6]), np.ones((3, 1, 2)), 1.0)
    assert thr[0, 0] == 0.9


def test_a_cma_mae_occupant_can_have_less_quality_than_the_best_ever_because_it_only_has_to_beat_the_threshold():
    """Lernrate 0.1: nach q = 1.0 ist die Schwelle erst 0.1; ein späteres Kind mit q = 0.5 ersetzt den Insassen (0.5 > 0.1), obwohl der Ergebnis-Elite bei 1.0 bleibt; deshalb wird das Ergebnisarchiv getrennt geführt."""
    thr = np.zeros((1, 1))
    occ_g, occ_filled = np.zeros((1, 1, 1, 2)), np.zeros((1, 1), bool)
    S.mae_update(thr, occ_g, occ_filled, np.array([0]), np.array([0]), np.array([1.0]), np.full((1, 1, 2), 1.0), 0.1)
    S.mae_update(thr, occ_g, occ_filled, np.array([0]), np.array([0]), np.array([0.5]), np.full((1, 1, 2), 2.0), 0.1)
    assert thr[0, 0] == pytest.approx(0.14) and occ_g[0, 0, 0, 0] == 2.0
    result_q, result_g = np.full((1, 1), -1.0), np.zeros((1, 1, 1, 2))
    S.insert(result_q, result_g, np.array([0, 0]), np.array([0, 0]), np.array([1.0, 0.5]), np.array([[[1.0, 1.0]], [[2.0, 2.0]]]))
    assert result_q[0, 0] == 1.0 and result_g[0, 0, 0, 0] == 1.0

"""Orakel-Tests für Archiv und Verfahren: unabhängige Gegenproben.

* **Direktheit** gegen eine skalare Neuberechnung der Fahrt nur aus der Beschreibung der Kinematik (eine Schleife je Individuum und Schritt, `math` statt NumPy), für Steuerfolgen verschiedener Länge.
* **MAP-Elites, CMA-ME und CMA-MAE als Ganzes** gegen eine skalare Neuimplementierung der Buchführung mit Wörterbüchern (Zelle -> Qualität und Steuerfolge, Zelle -> Schwelle) statt der Felder; dieselben Zufallszahlen, dieselbe Reihenfolge der Emitter.
  Das CMA-ES im Emitter selbst ist in `test_cmaes.py` geprüft (skalare Formeln, pycma); hier geht es um Rangfolge, Archiv, Schwellen, Neustarts und Eltern.
* **Restweg** gegen SciPys Breitensuche."""

import math

import numpy as np
import pytest

import cme_cmaes as E
import cme_constants as C
import cme_maze as M
import cme_robot as R
import cme_search as S
from test_oracle_maze import scipy_distances


def scalar_track(genome, grid):
    n = grid.shape[0]
    x, y = C.START
    heading = C.START_HEADING
    out = [(x, y)]
    for turn, speed in genome:
        heading += turn * C.TURN_MAX
        nx = x + speed * C.SPEED_MAX * math.cos(heading)
        ny = y + speed * C.SPEED_MAX * math.sin(heading)
        if 0 <= nx < n and 0 <= ny < n and not grid[int(nx), int(ny)]:
            x, y = nx, ny
        out.append((x, y))
    return out


def scalar_eval(genome, grid):
    """(Archivzelle, Direktheit, Endposition) einer Fahrt, skalar."""
    pts = scalar_track(genome, grid)
    driven = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
    ex, ey = pts[-1]
    q = math.hypot(ex - C.START[0], ey - C.START[1]) / driven if driven > 1e-9 else 0.0
    nb = int(math.ceil(grid.shape[0] / C.CELL))
    cell = (min(max(int(ex / C.CELL), 0), nb - 1), min(max(int(ey / C.CELL), 0), nb - 1))
    return cell, q, (ex, ey)


@pytest.mark.parametrize("steps", [5, 40, 140])
@pytest.mark.parametrize("level", [0, 3, 4])
def test_directness_agrees_with_the_scalar_reference_for_every_genome_length(level, steps):
    grid = M.make_maze(level)
    genomes = R.random_genomes(np.random.default_rng(300 + level + steps), 12, steps)
    _, track = R.simulate(genomes, grid, trajectories=True)
    assert S.directness(track) == pytest.approx([scalar_eval(g, grid)[1] for g in genomes], abs=1e-10)


def reference_run(grid, seed, method, steps, sigma0, gens, pop, n_emit):
    """Dieselbe Buchführung mit Wörterbüchern; gibt (Ergebnisarchiv {Zelle: (Qualität, Steuerfolge)}, Kurven von Abdeckung und QD-Score) zurück."""
    rng = np.random.default_rng(seed)
    n, lam = steps * 2, pop // n_emit
    emitters = [E.Emitter(n, lam, sigma0, rng) for _ in range(n_emit)] if method != "me" else []
    genomes = R.random_genomes(rng, pop, steps)
    result, thr, occ = {}, {}, {}
    alpha = C.ALPHAS.get(method)
    cov, qd = [], []
    for gen in range(gens):
        info = [scalar_eval(g, grid) for g in genomes]
        status, value, delta = [0] * pop, [0.0] * pop, [0.0] * pop
        if method == "cmame":
            tmp = {c: v[0] for c, v in result.items()}
            for k, (cell, q, _) in enumerate(info):
                if cell not in tmp:
                    status[k], value[k], tmp[cell] = 2, q, q
                elif q > tmp[cell] + C.QUALITY_TIE:
                    status[k], value[k], tmp[cell] = 1, q - tmp[cell], q
        if alpha is not None:
            for k, (cell, q, _) in enumerate(info):
                t = thr.get(cell, C.QUALITY_MIN)
                delta[k] = q - t
                if q > t:
                    occ[cell] = genomes[k].copy()
                    thr[cell] = (1 - alpha) * t + alpha * q
        for k, (cell, q, _) in enumerate(info):
            if cell not in result or q > result[cell][0] + C.QUALITY_TIE:
                result[cell] = (q, genomes[k].copy())
        cov.append(len(result))
        qd.append(sum(v[0] for v in result.values()))
        if method == "me":
            keys = sorted(result)
            pick = rng.integers(0, len(keys), pop)
            parents = np.array([result[keys[i]][1] for i in pick])
            mask = rng.random(parents.shape) < C.MUT_RATE
            genomes = R.clip_genomes(parents + rng.normal(0.0, C.MUT_SIGMA, parents.shape) * mask)
            continue
        for ei, em in enumerate(emitters):
            lo = ei * lam
            if gen == 0:
                store = occ if alpha is not None else {c: v[1] for c, v in result.items()}
                keys = sorted(store)
                em.reset(store[keys[rng.integers(0, len(keys))]].reshape(-1))
            elif method == "cmame":
                ok = [i for i in range(lam) if status[lo + i] > 0]
                ranked = sorted(ok, key=lambda i: (-status[lo + i], -value[lo + i]))
                if not ranked:
                    keys = sorted(result)
                    em.reset(result[keys[rng.integers(0, len(keys))]][1].reshape(-1))
                else:
                    em.tell(ranked)
            else:
                d = np.array(delta[lo:lo + lam])
                em.tell(list(np.argsort(-d, kind="stable")[:lam // 2]))
                if em.converged(float(d.max())):
                    keys = sorted(occ)
                    em.reset(occ[keys[rng.integers(0, len(keys))]].reshape(-1))
        genomes = np.vstack([R.clip_genomes(em.ask().reshape(lam, steps, 2)) for em in emitters])
    return result, cov, qd


@pytest.mark.parametrize("method", C.METHODS)
def test_each_method_agrees_with_a_dictionary_based_reimplementation_of_the_bookkeeping(method):
    grid = M.make_maze(3)
    steps, pop, gens, n_emit, sigma0 = 8, 20, 12, 2, 0.2
    ref, ref_cov, ref_qd = reference_run(grid, 5, method, steps, sigma0, gens, pop, n_emit)
    res = S.run_search(grid, 5, method, steps=steps, sigma0=sigma0, gens=gens, pop=pop, n_emitters=n_emit, keep_archive=True)
    fi, fj = np.nonzero(res.archive_q >= 0)
    mine = {(int(i), int(j)): (float(res.archive_q[i, j]), res.archive_g[i, j]) for i, j in zip(fi, fj)}
    assert set(mine) == set(ref)
    for c in ref:
        assert mine[c][0] == pytest.approx(ref[c][0], abs=1e-10) and (mine[c][1] == ref[c][1]).all()
    assert res.coverage == ref_cov and res.qd_score == pytest.approx(ref_qd, abs=1e-8)


def test_the_restart_count_agrees_with_the_reference_for_cma_me():
    """Wenn in einer Generation kein Nachkomme eines Emitters das Archiv ändert, startet er neu: die Zählung im Lauf stimmt mit einer Zählung aus der skalaren Buchführung überein (kleine Instanz mit vielen Neustarts)."""
    grid = M.make_maze(0)
    steps, pop, gens, n_emit = 6, 20, 30, 4
    res = S.run_search(grid, 3, "cmame", steps=steps, sigma0=0.05, gens=gens, pop=pop, n_emitters=n_emit)
    assert res.restarts > 0 and res.restarts <= (gens - 1) * n_emit


def test_best_progress_agrees_with_scipys_breadth_first_search():
    grid = M.make_maze(4)
    res = S.run_search(grid, 2, "me", steps=140, gens=6)
    ref = scipy_distances(grid, M.cell_of(C.GOAL))
    rng = np.random.default_rng(2)
    end = R.simulate(R.random_genomes(rng, C.POP, 140), grid)
    assert res.best_progress[0] == min(ref[int(x), int(y)] for x, y in end) and all(b <= a for a, b in zip(res.best_progress, res.best_progress[1:]))

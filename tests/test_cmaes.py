"""CMA-ES-Emitter: Aktualisierung von Hand an einem 2D-Beispiel gegen eine skalare Neuberechnung nach Hansens Tutorial, gegen pycma (`cma`) mit denselben Stichproben, Konvergenz auf Kugel und Ellipsoid,
Zurücksetzen, Konditionszahl, Abbruchkriterium."""

import math

import numpy as np
import pytest

import cme_cmaes as E


def scalar_tell(m, sigma, C, pc, ps, y, ranked, n, age):
    """Ein Aktualisierungsschritt nach Hansen (2016) in reinen Python-Listen, ohne NumPy: gibt (m, sigma, C, pc, ps) zurück. `C` ist die Kovarianzmatrix vor dem Schritt (hier mit B = I, D = 1 angenommen, also C^(-1/2) = I)."""
    mu = len(ranked)
    w = [math.log(mu + 0.5) - math.log(i) for i in range(1, mu + 1)]
    sw = sum(w)
    w = [x / sw for x in w]
    mueff = 1 / sum(x * x for x in w)
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chi = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
    ysel = [y[i] for i in ranked]
    yw = [sum(w[k] * ysel[k][j] for k in range(mu)) for j in range(n)]
    m_new = [m[j] + sigma * yw[j] for j in range(n)]
    ps_new = [(1 - cs) * ps[j] + math.sqrt(cs * (2 - cs) * mueff) * yw[j] for j in range(n)]
    hsig = 1.0 if math.sqrt(sum(x * x for x in ps_new)) / math.sqrt(1 - (1 - cs) ** (2 * age)) / chi < 1.4 + 2 / (n + 1) else 0.0
    pc_new = [(1 - cc) * pc[j] + hsig * math.sqrt(cc * (2 - cc) * mueff) * yw[j] for j in range(n)]
    C_new = [[(1 - c1 - cmu) * C[a][b] + c1 * (pc_new[a] * pc_new[b] + (1 - hsig) * cc * (2 - cc) * C[a][b]) + cmu * sum(w[k] * ysel[k][a] * ysel[k][b] for k in range(mu)) for b in range(n)] for a in range(n)]
    sigma_new = sigma * math.exp((cs / damps) * (math.sqrt(sum(x * x for x in ps_new)) / chi - 1))
    return m_new, sigma_new, C_new, pc_new, ps_new


def test_one_update_step_in_two_dimensions_agrees_with_the_scalar_formulas():
    em = E.Emitter(2, 4, 0.5, np.random.default_rng(1))
    em.reset(np.array([1.0, -2.0]))
    em.y = np.array([[0.5, 1.0], [-1.0, 0.2], [1.5, -0.5], [0.1, 0.1]])
    em.tell([2, 0, 3])
    m, sigma, C, pc, ps = scalar_tell([1.0, -2.0], 0.5, [[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0], [0.0, 0.0], em.y.tolist(), [2, 0, 3], 2, 1)
    assert em.m == pytest.approx(m, abs=1e-12) and em.sigma == pytest.approx(sigma, abs=1e-12) and em.pc == pytest.approx(pc, abs=1e-12) and em.ps == pytest.approx(ps, abs=1e-12)
    assert em.C == pytest.approx(np.array(C), abs=1e-12) and em.age == 1


def test_one_update_step_by_hand_with_a_single_parent():
    """Ein einziger Elter bekommt das Gewicht 1: der neue Mittelwert ist m + sigma · y; mueff = 1."""
    em = E.Emitter(2, 4, 0.5, np.random.default_rng(1))
    em.reset(np.array([0.0, 0.0]))
    em.y = np.array([[2.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    em.tell([0])
    assert em.m == pytest.approx([1.0, 0.0])
    n, mueff = 2, 1.0
    cs = (mueff + 2) / (n + mueff + 5)
    assert em.ps == pytest.approx([math.sqrt(cs * (2 - cs) * mueff) * 2.0, 0.0])


def test_ask_gives_mean_plus_sigma_times_the_standard_normal_draws_for_the_initial_distribution():
    em = E.Emitter(3, 5, 0.25, np.random.default_rng(7))
    em.reset(np.array([1.0, 2.0, 3.0]))
    x = em.ask()
    z = np.random.default_rng(7).standard_normal((5, 3))
    assert x == pytest.approx(np.array([1.0, 2.0, 3.0]) + 0.25 * z) and em.y == pytest.approx(z)


def test_reset_restores_the_initial_distribution():
    em = E.Emitter(4, 8, 0.3, np.random.default_rng(2))
    em.reset(np.zeros(4))
    em.ask()
    em.tell([0, 1, 2, 3])
    em.reset(np.ones(4))
    assert (em.m == 1).all() and em.sigma == 0.3 and (em.C == np.eye(4)).all() and (em.pc == 0).all() and (em.ps == 0).all() and em.age == 0 and em.condition() == pytest.approx(1.0)


def test_the_condition_number_is_one_for_the_start_and_grows_when_the_search_learns_an_ellipsoid():
    n = 10
    em = E.Emitter(n, 20, 0.5, np.random.default_rng(3))
    em.reset(np.full(n, 1.0))
    scale = np.logspace(0, 3, n)
    assert em.condition() == pytest.approx(1.0)
    for _ in range(250):
        x = em.ask()
        f = ((x * scale) ** 2).sum(axis=1)
        em.tell(list(np.argsort(f)[:10]))
    assert em.condition() > 100


def _run_function(f, n=10, lam=20, sigma0=0.5, evals=20000, seed=5, x0=1.0):
    em = E.Emitter(n, lam, sigma0, np.random.default_rng(seed))
    em.reset(np.full(n, x0))
    best = np.inf
    used = 0
    while used < evals and best > 1e-10:
        x = em.ask()
        fx = f(x)
        used += lam
        best = min(best, float(fx.min()))
        em.tell(list(np.argsort(fx)[:lam // 2]))
    return best, used


def test_it_converges_on_the_sphere_and_on_an_ellipsoid():
    sphere = lambda x: (x ** 2).sum(axis=1)
    ellipsoid = lambda x: ((x * np.logspace(0, 3, x.shape[1])) ** 2).sum(axis=1)
    best_s, used_s = _run_function(sphere, evals=6000)
    best_e, used_e = _run_function(ellipsoid, evals=30000)
    assert best_s < 1e-10 and used_s < 6000 and best_e < 1e-10 and used_e < 30000


def test_the_mean_agrees_exactly_and_sigma_and_covariance_closely_with_pycma_when_fed_the_same_samples():
    """pycma (Paket `cma`) als unabhängige Referenz. pycma weicht in zwei Lernraten vom Tutorial ab (Pfadlänge cs mit Nenner N + mueff + 3 statt + 5, Rang-mu-Rate cmu); sie werden hier auf die Werte des Tutorials gesetzt.
    Mit denselben Stichproben und derselben Rangfolge stimmt dann der Mittelwert auf 1e-12 überein (gewichtete Rekombination), Schrittweite und Kovarianz bleiben innerhalb von 3 % beziehungsweise 0.05 über 6 Schritte."""
    cma = pytest.importorskip("cma")
    n, lam = 6, 12
    mu = lam // 2
    opts = {"popsize": lam, "CMA_active": False, "CMA_diagonal": False, "verbose": -9, "seed": 11, "tolx": 0, "tolfun": 0, "tolfunhist": 0, "tolstagnation": 10 ** 9, "CMA_mu": mu}
    es = cma.CMAEvolutionStrategy([1.0] * n, 0.5, opts)
    w = np.array([math.log(mu + 0.5) - math.log(i) for i in range(1, mu + 1)])
    w = w / w.sum()
    mueff = 1 / np.sum(w ** 2)
    cs = (mueff + 2) / (n + mueff + 5)
    es.adapt_sigma.cs, es.adapt_sigma.damps = cs, 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    es.sp.cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    ours = E.Emitter(n, lam, 0.5, np.random.default_rng(0))
    ours.reset(np.full(n, 1.0))
    f = lambda x: float(sum((i + 1) * v * v for i, v in enumerate(x)))
    for _ in range(6):
        xs = es.ask()
        fx = [f(x) for x in xs]
        ours.y = (np.array(xs) - ours.m) / ours.sigma
        ours.tell(list(np.argsort(fx)[:mu]))
        es.tell(xs, fx)
        assert ours.m == pytest.approx(np.array(es.mean), abs=1e-12)
        assert ours.sigma == pytest.approx(es.sigma, rel=0.03) and np.abs(ours.C - np.array(es.C)).max() < 0.05


def test_the_number_of_evaluations_to_converge_on_an_ellipsoid_is_comparable_to_pycma():
    """Mit denselben Einstellungen (10 Dimensionen, 20 Nachkommen, Ellipsoid mit Konditionszahl 10^6, Start bei 1, Schrittweite 0.5, Ziel 1e-10) braucht unser Emitter im Mittel über 3 Läufe zwischen dem 0.5- und 2-fachen der Evaluationen von pycma."""
    cma = pytest.importorskip("cma")
    ellipsoid = lambda x: ((x * np.logspace(0, 3, x.shape[1])) ** 2).sum(axis=1)
    ours, theirs = [], []
    for seed in (1, 2, 3):
        best, used = _run_function(ellipsoid, seed=seed, evals=60000)
        assert best < 1e-10
        ours.append(used)
        es = cma.CMAEvolutionStrategy([1.0] * 10, 0.5, {"popsize": 20, "CMA_active": False, "verbose": -9, "seed": seed, "ftarget": 1e-10, "tolx": 0, "tolfun": 0, "tolfunhist": 0})
        es.optimize(lambda x: float(((np.array(x) * np.logspace(0, 3, 10)) ** 2).sum()))
        theirs.append(es.result.evaluations)
    assert 0.5 < np.mean(ours) / np.mean(theirs) < 2.0


def test_tell_with_all_offspring_ranked_uses_log_weights_summing_to_one():
    em = E.Emitter(3, 6, 0.4, np.random.default_rng(4))
    em.reset(np.zeros(3))
    em.y = np.eye(3)[[0, 1, 2, 0, 1, 2]].astype(float)
    em.tell(list(range(6)))
    w = np.array([math.log(6.5) - math.log(i) for i in range(1, 7)])
    w = w / w.sum()
    expected = 0.4 * (w @ em.y)
    assert em.m == pytest.approx(expected, abs=1e-12)


def test_converged_by_small_step_by_high_condition_and_by_stagnation():
    em = E.Emitter(3, 6, 0.4, np.random.default_rng(1))
    em.reset(np.zeros(3))
    assert not em.converged(1.0)
    em.sigma = 1e-9
    assert em.converged(1.0)
    em.reset(np.zeros(3))
    em.C = np.diag([1.0, 1.0, 1e-14])
    em.B, em.D = np.eye(3), np.sqrt(np.array([1.0, 1.0, 1e-14]))
    assert em.converged(1.0)
    em.reset(np.zeros(3))
    flags = [em.converged(v) for v in [5.0] * 10 + [1.0] * 30]
    assert not any(flags[:10]) and any(flags[10:])
    em.reset(np.zeros(3))
    assert not any(em.converged(float(v)) for v in range(60))                    # steigende beste Werte: keine Stagnation

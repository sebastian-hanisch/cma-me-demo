"""Auswertung: Täuschungsmaße je Stufe (exakt), Lösbarkeitsnachweis, Live-Paar (MAP-Elites neben einem CMA-Verfahren), Lesen und Aggregieren der vorgerechneten Studie, Mann-Whitney-Test.
Die Studie (Dimensions-Sweep im freien Feld, Labyrinth bei nativer Dimension, Sweep der Anfangsschrittweite) wird beim Bau gerechnet (`generate_precomputed.py`) und hier nur gelesen."""

import json
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

import cme_constants as C
import cme_maze as M
import cme_robot as R
import cme_search as S

try:
    from threadpoolctl import threadpool_limits
except ImportError:                                  # ohne threadpoolctl läuft alles weiter, nur ohne Begrenzung der BLAS-Threads
    def threadpool_limits(*args, **kwargs):
        return nullcontext()

PRECOMPUTED_PATH = Path(__file__).resolve().parent / "precomputed_sweep.json"


def level_metrics(level):
    """Exakte Täuschungsmaße der Stufe: kürzester Weg (Zellen), Luftlinie Start–Ziel, Umwegfaktor, Fitness-Distanz-Korrelation, lokale Minima (Zahl), Abstand des tiefsten Minimums zum Ziel, freie Zellen."""
    g = M.make_maze(level)
    path = M.shortest_path(g)
    return {"path": len(path) - 1, "straight": float(np.hypot(C.START[0] - C.GOAL[0], C.START[1] - C.GOAL[1])), "detour": M.detour_ratio(g), "fdc": M.fitness_distance_correlation(g),
            "minima": len(M.local_minima(g)), "trap": M.trap_distance(g), "free": int((~g).sum())}


def solvability(level):
    """Lösbarkeitsnachweis: eine aus dem kürzesten Pfad abgeleitete Steuerfolge fährt in diesem Labyrinth ins Ziel. Gibt (Abstand am Ende, benötigte Schritte) zurück."""
    g = M.make_maze(level)
    genome = R.controller_from_path(M.shortest_path(g))
    end = R.simulate(genome[None], g)
    return float(R.distance_to_goal(end)[0]), int((np.abs(genome).sum(axis=1) > 0).sum())


def live_run(level, gens, seed, method, steps, sigma):
    """Ein Lauf mit dem Verfahren `method`; das Ergebnisarchiv wird gespeichert und seine Fahrten werden neu gerechnet. Gibt (Lauf, Qualität je Elite, Fahrten, Zellen) zurück."""
    g = M.make_maze(level)
    with threadpool_limits(limits=1, user_api="blas"):          # die Matrizen der Emitter sind klein; viele BLAS-Threads bremsen auf geteilten Rechnern (Streamlit Cloud) die Eigenzerlegung um ein Vielfaches
        res = S.run_search(g, seed, method, steps=steps, sigma0=sigma, gens=gens, keep_archive=True)
    q, tracks, cells = S.elite_tracks(g, res)
    return res, q, tracks, cells


def live_pair(level, gens, seed, steps, sigma, method):
    """Derselbe Seed, dieselbe Stufe und Genomlänge: MAP-Elites (Bezug) und das gewählte CMA-Verfahren. Gibt die beiden `live_run`-Ergebnisse zurück."""
    return live_run(level, gens, seed, "me", steps, sigma), live_run(level, gens, seed, method, steps, sigma)


@lru_cache(maxsize=1)
def load_precomputed():
    return json.loads(PRECOMPUTED_PATH.read_text(encoding="utf-8"))


def success_count(runs, budget=None):
    """Zahl der Läufe, die den Zielkreis erreichten (höchstens `budget` Evaluationen, wenn angegeben)."""
    return sum(1 for r in runs if r is not None and (budget is None or r <= budget))


def median_solved(runs):
    """Median der Evaluationen bis zum Erfolg über die erfolgreichen Läufe (None, wenn kein Lauf erfolgreich war)."""
    ok = [r for r in runs if r is not None]
    return float(np.median(ok)) if ok else None


def mannwhitney_p(values_a, values_b):
    """Zweiseitiger Mann-Whitney-Test auf zwei Zahlenlisten (etwa den QD-Score je Lauf)."""
    return float(mannwhitneyu(values_a, values_b, alternative="two-sided").pvalue)


def case_info(pre, case):
    return pre["cases"][case]


def study_cell(pre, case, method):
    return pre["cases"][case]["methods"][method]


def sigma_cell(pre, case, method, sigma):
    return pre["sigma"][case][method][str(sigma)]

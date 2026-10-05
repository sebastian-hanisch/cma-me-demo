"""Rechnet die Studie vor (Build-Zeit, nicht in der App): `python generate_precomputed.py [Prozesse]` schreibt `precomputed_sweep.json`.

  cases   MAP-Elites, CMA-ME und CMA-MAE (Lernraten 0.01 / 0.1 / 0.5) mit Anfangsschrittweite 0.2, je 100 Nachkommen je Generation (5 Emitter × 20), 400 Generationen (40 000 Evaluationen), je 20 Läufe:
          d10 … d140   freies Feld, Steuerfolge 10 / 20 / 40 / 80 / 140 Schritte (20 bis 280 Dimensionen)
          l3, l4       Stufe 3 und 4 bei 140 Schritten
          je Zelle: Evaluationen bis zum ersten Erfolg (None = nie), wahrer Restweg, beste Direktheit, Abdeckung, QD-Score und Zahl der Emitter-Neustarts am Ende, mittlerer Verlauf von Abdeckung und QD-Score,
          mittlerer Zehnerlogarithmus der Konditionszahl der Emitter-Kovarianz zu den Meilensteinen
  sigma   Anfangsschrittweite 0.05 / 0.1 / 0.2 / 0.4 für CMA-ME und CMA-MAE (Lernrate 0.1) im freien Feld bei 40 und 140 Schritten, je 20 Läufe: Abdeckung und QD-Score am Ende"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import cme_constants as C
import cme_evaluation as E
import cme_maze as M
import cme_search as S


def _case_task(args):
    case, method, seed = args
    level, steps = C.CASES[case]
    res = S.run_search(M.make_maze(level), seed, method, steps=steps, sigma0=C.DEFAULT_SIGMA, gens=C.STUDY_GENS)
    return case, method, seed, res.first_solved, res.best_progress[-1], res.best_quality[-1], res.coverage, res.qd_score, res.restarts, res.conditions, res.total


def _sigma_task(args):
    case, method, sigma, seed = args
    level, steps = C.SIGMA_CASES[case]
    res = S.run_search(M.make_maze(level), seed, method, steps=steps, sigma0=sigma, gens=C.STUDY_GENS)
    return case, method, sigma, seed, res.coverage[-1], res.qd_score[-1]


def _curve(rows, index):
    return [round(float(x), 3) for x in np.mean([r[index] for r in rows], axis=0)]


def main(workers):
    t0 = time.time()
    case_tasks = [(case, m, s) for case in C.CASES for m in C.METHODS for s in range(C.STUDY_SEEDS)]
    sigma_tasks = [(case, m, sg, s) for case in C.SIGMA_CASES for m in C.SIGMA_METHODS for sg in C.SIGMAS for s in range(C.STUDY_SEEDS)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        case_res = list(ex.map(_case_task, case_tasks, chunksize=2))
        print(f"Fälle fertig nach {time.time() - t0:.0f} s", flush=True)
        sigma_res = list(ex.map(_sigma_task, sigma_tasks, chunksize=2))
    cases = {}
    for case, (level, steps) in C.CASES.items():
        methods = {}
        for m in C.METHODS:
            rows = sorted([r for r in case_res if r[0] == case and r[1] == m], key=lambda r: r[2])
            conds = {str(g): round(float(np.mean([r[9][g] for r in rows])), 4) for g in C.MILESTONES} if m != "me" else {}
            methods[m] = {"runs": [r[3] for r in rows], "progress": [r[4] for r in rows], "best_quality": [round(r[5], 6) for r in rows], "coverage": [r[6][-1] for r in rows],
                          "qd": [round(r[7][-1], 6) for r in rows], "restarts": [r[8] for r in rows], "mean_coverage": _curve(rows, 6), "mean_qd": _curve(rows, 7), "log_condition": conds}
        total = next(r[10] for r in case_res if r[0] == case)
        cases[case] = {"level": level, "steps": steps, "seeds": C.STUDY_SEEDS, "metrics": E.level_metrics(level), "total": total, "methods": methods}
    sigma = {}
    for case in C.SIGMA_CASES:
        sigma[case] = {}
        for m in C.SIGMA_METHODS:
            sigma[case][m] = {}
            for sg in C.SIGMAS:
                rows = sorted([r for r in sigma_res if r[0] == case and r[1] == m and r[2] == sg], key=lambda r: r[3])
                sigma[case][m][str(sg)] = {"coverage": [r[4] for r in rows], "qd": [round(r[5], 6) for r in rows]}
    data = {"budget": C.STUDY_BUDGET, "cases": cases, "sigma": sigma}
    E.PRECOMPUTED_PATH.write_text(json.dumps(data), encoding="utf-8")
    print(f"fertig in {time.time() - t0:.0f} s -> {E.PRECOMPUTED_PATH}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else min(14, os.cpu_count() or 1))

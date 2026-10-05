"""MAP-Elites, CMA-ME und CMA-MAE mit demselben Archiv über dem Verhaltensraum.

**Archiv (wie Stück 5):** die Endposition der Fahrt, in Blöcke der Kantenlänge 2 Zellen unterteilt (144 Archivzellen mit freier Fläche); **Qualität** = Direktheit der Fahrt (Luftlinie von Start zu Ende geteilt durch die gefahrene Weglänge,
0 bis 1). Das **Ergebnisarchiv** hält je Zelle die beste Fahrt aller bewerteten Steuerfolgen (ein Kind ersetzt den Elite nur bei strikt höherer Qualität); alle Kennzahlen kommen daraus, bei allen Verfahren gleich definiert:
Abdeckung (belegte Zellen), QD-Score (Summe der Qualität über die belegten Zellen), beste Qualität.

**MAP-Elites:** `pop` Eltern gleichverteilt aus den belegten Zellen des Ergebnisarchivs, Mutation Gauß 0.25 auf 15 % der Gene (wie Stück 5).

**CMA-ME** (Fontaine u. a. 2020), Verbesserungs-Emitter: `N_EMITTERS` Emitter mit je `pop // N_EMITTERS` Nachkommen. Jeder Nachkomme wird ins Ergebnisarchiv eingetragen; er heißt *neue Zelle* (die Zelle war leer) oder *Verbesserung* (er
übertrifft den Elite um mehr als `C.QUALITY_TIE`; Verbesserung = Qualität − bisherige Qualität). Der Emitter ordnet nur die eingetragenen Nachkommen: erst alle mit neuer Zelle (nach Qualität), dann die Verbesserungen (nach Verbesserung), und aktualisiert damit sein CMA-ES.
Wurde in einer Generation kein Nachkomme eingetragen, startet der Emitter neu: bei einem zufälligen Elite, Schrittweite σ₀, Kovarianz Einheitsmatrix.

**CMA-MAE** (Fontaine und Nikolaidis 2023): ein zweites Archiv der **Schwellen** je Zelle (Start: kleinste Qualität 0). Für jeden Nachkommen: Verbesserung Δ = Qualität − Schwelle; ist die Qualität größer als die Schwelle, ersetzt er den
Insassen des Emitter-Archivs und die Schwelle steigt: `t ← (1 − α) t + α q` (Lernrate α). Der Emitter ordnet **alle** Nachkommen nach Δ und aktualisiert sein CMA-ES mit der besseren Hälfte; er startet neu, wenn sein CMA-ES konvergiert ist (`Emitter.converged`),
bei einem zufälligen Insassen des Emitter-Archivs. Das Emitter-Archiv (Schwellen und Insassen) dient nur den Emittern; gemessen wird das Ergebnisarchiv.

Zufall nur über `numpy.random.default_rng(seed)` (PCG64, plattformstabil)."""

from dataclasses import dataclass, field

import numpy as np

import cme_cmaes as E
import cme_constants as C
import cme_maze as M
import cme_robot as R


@dataclass
class SearchResult:
    best_distance: list                  # kleinster Abstand zum Ziel je Generation
    best_progress: list                  # kleinster wahrer Restweg (Zellen) aller bisher besuchten Zellen nach jeder Generation
    coverage: list                       # belegte Zellen des Ergebnisarchivs nach jeder Generation
    qd_score: list                       # Summe der Qualität über die belegten Zellen nach jeder Generation
    best_quality: list                   # beste Qualität nach jeder Generation
    total: int                           # Zahl der Archivzellen mit freier Fläche
    first_solved: object                 # Zahl der Evaluationen bis zum ersten Individuum im Zielkreis (None: nie)
    restarts: int = 0                    # Zahl der Emitter-Neustarts im ganzen Lauf
    conditions: dict = field(default_factory=dict)   # Generation -> mittlere Konditionszahl (Zehnerlogarithmus) der Emitter-Kovarianzen zu den Meilensteinen C.MILESTONES (leer bei MAP-Elites)
    archive_q: object = None             # Qualität je Zelle (−1 = leer) des Ergebnisarchivs, Form (12, 12)
    archive_g: object = None             # Steuerfolge je Zelle, Form (12, 12, Schritte, 2)
    evaluations: int = 0


def directness(track):
    """Qualität jeder Fahrt: Luftlinie Start–Ende geteilt durch die gefahrene Weglänge (0, wenn der Roboter gar nicht gefahren ist). `track` hat die Form (P, Schritte + 1, 2)."""
    driven = np.linalg.norm(np.diff(track, axis=1), axis=2).sum(axis=1)
    straight = np.linalg.norm(track[:, -1, :] - np.array(C.START), axis=1)
    return np.where(driven > 1e-9, straight / np.maximum(driven, 1e-9), 0.0)


def n_blocks():
    return int(np.ceil(C.N / C.CELL))


def cell_indices(end):
    """Archivzelle (i, j) jeder Endposition."""
    idx = np.clip((np.asarray(end) / C.CELL).astype(int), 0, n_blocks() - 1)
    return idx[:, 0], idx[:, 1]


def free_blocks(grid):
    """Bool-Raster: Archivzellen, die mindestens eine freie Zelle des Labyrinths enthalten."""
    nb = n_blocks()
    out = np.zeros((nb, nb), bool)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not grid[i, j]:
                out[i // C.CELL, j // C.CELL] = True
    return out


def insert(quality, elites, ci, cj, q, genomes):
    """Trägt Kinder der Reihe nach ins Archiv ein: ein Kind ersetzt den Elite seiner Zelle, wenn es dort um mehr als `C.QUALITY_TIE` mehr Qualität hat (oder die Zelle leer ist, Qualität −1); gleiche Qualität ersetzt nicht."""
    for kid in range(len(q)):
        if q[kid] > quality[ci[kid], cj[kid]] + C.QUALITY_TIE:
            quality[ci[kid], cj[kid]] = q[kid]
            elites[ci[kid], cj[kid]] = genomes[kid]


def archive_metrics(quality):
    """Kennzahlen eines Archivs (Qualität je Zelle, −1 = leer): (Abdeckung, QD-Score)."""
    filled = quality >= 0
    return int(filled.sum()), float(quality[filled].sum())


def status_and_value(quality, ci, cj, q):
    """Für CMA-ME: spielt das Eintragen der Kinder der Reihe nach auf einer Kopie von `quality` durch (das Archiv selbst bleibt unverändert) und gibt Status (2 = neue Zelle, 1 = Verbesserung, 0 = nichts) und Wert (Qualität bei neuer Zelle,
    Verbesserung sonst) je Kind zurück. Ein zweites Kind derselben Zelle sieht also den Eintrag des ersten."""
    tmp = quality.copy()
    status, value = np.zeros(len(q), int), np.zeros(len(q))
    for kid in range(len(q)):
        c = (ci[kid], cj[kid])
        if tmp[c] < 0:
            status[kid], value[kid] = 2, q[kid]
            tmp[c] = q[kid]
        elif q[kid] > tmp[c] + C.QUALITY_TIE:
            status[kid], value[kid] = 1, q[kid] - tmp[c]
            tmp[c] = q[kid]
    return status, value


def rank_improvements(status, value):
    """CMA-ME: Reihenfolge (beste zuerst) der eingetragenen Nachkommen: erst neue Zellen, dann Verbesserungen, je absteigend nach Wert; Nachkommen ohne Eintrag fehlen."""
    ok = np.nonzero(status > 0)[0]
    return ok[np.lexsort((-value[ok], -status[ok]))]


def mae_update(thr, occ_g, occ_filled, ci, cj, q, genomes, alpha):
    """CMA-MAE: trägt die Kinder der Reihe nach ins Emitter-Archiv ein. Gibt Δ = Qualität − Schwelle (vor dem Eintrag) je Kind zurück. Ist die Qualität größer als die Schwelle, ersetzt das Kind den Insassen der Zelle und die Schwelle
    steigt auf (1 − α) · Schwelle + α · Qualität."""
    delta = np.zeros(len(q))
    for kid in range(len(q)):
        c = (ci[kid], cj[kid])
        delta[kid] = q[kid] - thr[c]
        if q[kid] > thr[c]:
            occ_g[c] = genomes[kid]
            occ_filled[c] = True
            thr[c] = (1 - alpha) * thr[c] + alpha * q[kid]
    return delta


def _random_elite(rng, filled, store):
    fi, fj = np.nonzero(filled)
    k = rng.integers(0, len(fi))
    return store[fi[k], fj[k]].reshape(-1)


def run_search(grid, seed, method, steps=C.STEPS, sigma0=C.DEFAULT_SIGMA, gens=C.STUDY_GENS, pop=C.POP, n_emitters=C.N_EMITTERS, keep_archive=False, stop_when_solved=False):
    """Ein Lauf des Verfahrens `method` (Schlüssel aus C.METHODS) mit Steuerfolgen der Länge `steps` (Dimension 2 · steps) und Anfangsschrittweite `sigma0` der Emitter."""
    if method not in C.METHODS:
        raise ValueError(f"unbekanntes Verfahren: {method}")
    rng = np.random.default_rng(seed)
    n = steps * 2
    nb = n_blocks()
    quality = np.full((nb, nb), -1.0)
    elites = np.zeros((nb, nb, steps, 2))
    is_mae = method in C.ALPHAS
    thr = np.full((nb, nb), C.QUALITY_MIN)
    occ_g = np.zeros((nb, nb, steps, 2))
    occ_filled = np.zeros((nb, nb), bool)
    true_path = M.bfs_distances(grid, M.cell_of(C.GOAL))
    lam = pop // n_emitters
    emitters = [E.Emitter(n, lam, sigma0, rng) for _ in range(n_emitters)] if method != "me" else []
    res = SearchResult(best_distance=[], best_progress=[], coverage=[], qd_score=[], best_quality=[], total=int(free_blocks(grid).sum()), first_solved=None)
    genomes = R.random_genomes(rng, pop, steps)
    best_q, best_p = -1.0, np.inf
    for gen in range(gens):
        end, track = R.simulate(genomes, grid, trajectories=True)
        dist = R.distance_to_goal(end)
        res.evaluations += pop
        if res.first_solved is None and dist.min() < C.SUCCESS_RADIUS:
            res.first_solved = gen * pop + int(np.argmax(dist < C.SUCCESS_RADIUS)) + 1
        idx = np.clip(end.astype(int), 0, grid.shape[0] - 1)
        best_p = min(best_p, float(true_path[idx[:, 0], idx[:, 1]].min()))
        q = directness(track)
        best_q = max(best_q, float(q.max()))
        ci, cj = cell_indices(end)
        if method == "cmame":
            status, value = status_and_value(quality, ci, cj, q)
        if is_mae:
            delta = mae_update(thr, occ_g, occ_filled, ci, cj, q, genomes, C.ALPHAS[method])
        insert(quality, elites, ci, cj, q, genomes)
        cov, qd = archive_metrics(quality)
        res.coverage.append(cov)
        res.qd_score.append(qd)
        res.best_distance.append(float(dist.min()))
        res.best_progress.append(best_p)
        res.best_quality.append(best_q)
        if stop_when_solved and res.first_solved is not None:
            break
        if method == "me":
            fi, fj = np.nonzero(quality >= 0)
            pick = rng.integers(0, len(fi), pop)
            parents = elites[fi[pick], fj[pick]]
            mask = rng.random(parents.shape) < C.MUT_RATE
            genomes = R.clip_genomes(parents + rng.normal(0.0, C.MUT_SIGMA, parents.shape) * mask)
            continue
        for ei, em in enumerate(emitters):
            sl = slice(ei * lam, (ei + 1) * lam)
            if gen == 0:
                em.reset(_random_elite(rng, occ_filled, occ_g) if is_mae else _random_elite(rng, quality >= 0, elites))
            elif method == "cmame":
                ranked = rank_improvements(status[sl], value[sl])
                if len(ranked) == 0:
                    em.reset(_random_elite(rng, quality >= 0, elites))
                    res.restarts += 1
                else:
                    em.tell(ranked)
            else:
                order = np.argsort(-delta[sl], kind="stable")
                em.tell(order[:lam // 2])
                if em.converged(float(delta[sl].max())):
                    em.reset(_random_elite(rng, occ_filled, occ_g))
                    res.restarts += 1
        if gen + 1 in C.MILESTONES:
            res.conditions[gen + 1] = float(np.mean([np.log10(em.condition()) for em in emitters]))
        genomes = np.vstack([R.clip_genomes(em.ask().reshape(lam, steps, 2)) for em in emitters])
    if keep_archive:
        res.archive_q, res.archive_g = quality, elites
    return res


def elite_tracks(grid, res):
    """Fahrten aller Elites des gespeicherten Ergebnisarchivs: (Qualität je Elite, Fahrten (E, Schritte + 1, 2), Zellen (E, 2))."""
    if res.archive_q is None:
        raise ValueError("kein Archiv gespeichert (keep_archive=True angeben)")
    fi, fj = np.nonzero(res.archive_q >= 0)
    genomes = res.archive_g[fi, fj]
    _, track = R.simulate(genomes, grid, trajectories=True)
    return res.archive_q[fi, fj], track, np.column_stack([fi, fj])

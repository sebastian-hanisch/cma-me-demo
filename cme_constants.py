"""Konstanten der CMA-ME-Demo: Labyrinth und Roboter wie in Stück 1 bis 5, Archiv wie Stück 5 (Zellgröße 2, Qualität Direktheit), Emitter-Verfahren, Genomlängen, Studien-Achsen, Regler, Voreinstellungen.
Längen in Zellen (Zellraster N × N, Zelle (i, j) belegt den Bereich [i, i + 1) × [j, j + 1))."""


def fmt_int(n):
    """Ganzzahl mit Leerzeichen als Tausendertrenner (10000 -> 10 000), wie in der README."""
    return f"{n:,}".replace(",", " ")


def fmt_med(x):
    """Median der Evaluationen: ganze Zahl mit Leerzeichen als Tausendertrenner, ein halber Wert behält seine Stelle (988.5, 1 308.5); None = kein Lauf war erfolgreich."""
    if x is None:
        return "–"
    return fmt_int(int(x)) if x == int(x) else f"{x:,.1f}".replace(",", " ")


N = 24                                       # Kantenlänge des Labyrinths in Zellen (Rand ist Wand)
START = (3.5, 3.5)                           # Mitte der Startzelle (3, 3)
GOAL = (3.5, 20.5)                           # Mitte der Zielzelle (3, 20)
SUCCESS_RADIUS = 2.0                         # Erfolg: Endposition höchstens so weit vom Ziel
START_HEADING = 0.7853981633974483           # Anfangsrichtung π/4
STEPS = 140                                  # Standardlänge der Steuerfolge (280 Dimensionen)
TURN_MAX = 0.6                               # Drehung je Schritt (rad), Gen ∈ [−1, 1]
SPEED_MAX = 0.7                              # Fahrstrecke je Schritt (Zellen), Gen ∈ [0, 1]

POP = 100                                    # Nachkommen je Generation (alle Emitter zusammen)
N_EMITTERS = 5                               # Emitter; je Emitter POP // N_EMITTERS = 20 Nachkommen
MUT_SIGMA = 0.25                             # MAP-Elites: Standardabweichung der Mutation je verändertem Gen
MUT_RATE = 0.15                              # MAP-Elites: Anteil der Gene, die je Kind mutieren

LEVELS = (0, 1, 2, 3, 4)
LEVEL_LABELS = {0: "0: freies Feld", 1: "1: Öffnung links", 2: "2: Öffnung in der Mitte", 3: "3: Öffnung rechts", 4: "4: Schlangenlinie"}
GAP = 3                                      # Breite einer Öffnung in der Wand (Zellen)

# Archiv wie Stück 5: Endposition in Blöcken der Kantenlänge CELL Zellen (144 Archivzellen), Qualität = Direktheit (Luftlinie Start–Ende geteilt durch gefahrene Weglänge, 0 bis 1)
CELL = 2
QUALITY_MIN = 0.0                            # kleinste mögliche Qualität: Anfangswert der Schwellen von CMA-MAE
QUALITY_TIE = 1e-12                          # Toleranz beim Ersetzen eines Elites: ein Kind muss um mehr als das besser sein (Gleichstände bis auf Rundung ersetzen nicht, auf jeder Plattform gleich)

# Verfahren: MAP-Elites (Bezug aus Stück 5), CMA-ME (Verbesserungs-Emitter) und CMA-MAE mit drei Lernraten alpha
METHODS = ("me", "cmame", "mae01", "mae10", "mae50")
ALPHAS = {"mae01": 0.01, "mae10": 0.1, "mae50": 0.5}
CMA_METHODS = ("cmame", "mae01", "mae10", "mae50")
METHOD_LABELS = {"me": "MAP-Elites (Bezug aus Stück 5)", "cmame": "CMA-ME", "mae01": "CMA-MAE, Lernrate 0.01", "mae10": "CMA-MAE, Lernrate 0.1", "mae50": "CMA-MAE, Lernrate 0.5"}
METHOD_GROUP = {"me": "me", "cmame": "cma", "mae01": "cma", "mae10": "cma", "mae50": "cma"}
GROUP_LABELS = {"me": "MAP-Elites", "cma": "CMA-Emitter"}

SIGMAS = (0.05, 0.1, 0.2, 0.4)               # Anfangsschrittweite σ₀ der Emitter
DEFAULT_SIGMA = 0.2
STEP_OPTIONS = (10, 20, 40, 80, 140)         # Längen der Steuerfolge (Dimension = 2 · Länge)
DEFAULT_STEPS = 40

STUDY_SEEDS = 20                             # Läufe je Zelle der Studie
STUDY_GENS = 400                             # Generationen je Lauf (bei POP = 100: 40 000 Evaluationen)
STUDY_BUDGET = POP * STUDY_GENS
MILESTONES = (10, 25, 50, 100, 200, 400)     # Generationen, zu denen die Konditionszahl der Emitter-Kovarianz festgehalten wird
# Fälle der vorgerechneten Studie: Schlüssel -> (Stufe, Länge der Steuerfolge)
DIM_CASES = {f"d{st}": (0, st) for st in STEP_OPTIONS}                     # Dimensions-Sweep im freien Feld
MAZE_CASES = {"l3": (3, STEPS), "l4": (4, STEPS)}                           # Labyrinth bei nativer Dimension
SIGMA_CASES = {"s40": (0, 40), "s140": (0, 140)}                            # σ₀-Sweep im freien Feld bei 40 und 140 Schritten
SIGMA_METHODS = ("cmame", "mae10")
CASES = {**DIM_CASES, **MAZE_CASES}
CASE_LABELS = {**{f"d{st}": f"freies Feld, {st} Schritte ({2 * st} Dimensionen)" for st in STEP_OPTIONS}, "l3": "Stufe 3 (140 Schritte)", "l4": "Stufe 4 (140 Schritte)"}

GENS_MIN, GENS_MAX, GENS_STEP, DEFAULT_GENS = 10, 200, 10, 100      # Regler: Generationen des Live-Laufs
DEFAULT_LEVEL = 0
SEED_MAX = 999999
DEFAULT_SEED = 35
DEFAULT_METHOD = "mae10"

PRESET_ORDER = ("Wenige Dimensionen", "Native Dimension", "Falle", "Zu kleine Schrittweite")


def _preset(level=DEFAULT_LEVEL, gens=DEFAULT_GENS, steps=DEFAULT_STEPS, sigma=DEFAULT_SIGMA, method=DEFAULT_METHOD):
    return {"level": level, "gens": gens, "seed": DEFAULT_SEED, "steps": steps, "sigma": sigma, "method": method}


PRESETS = {
    "Wenige Dimensionen": _preset(steps=20),
    "Native Dimension": _preset(steps=140),
    "Falle": _preset(level=3, steps=140, method="cmame"),
    "Zu kleine Schrittweite": _preset(sigma=0.05, method="cmame"),
}
# Zahlen aus der vorgerechneten Studie; tests/test_claims.py rechnet jede nach
PRESET_HELP = {
    "Wenige Dimensionen": "Freies Feld, 20 Schritte (40 Dimensionen), CMA-MAE mit Lernrate 0.1: QD-Score 70.67 gegen 51.90 bei MAP-Elites (+36 %; Mann-Whitney p < 0.001), belegte Zellen 70.90 gegen 52.15 von 144.",
    "Native Dimension": "140 Schritte (280 Dimensionen, die Länge der früheren Stücke): kein Vorsprung mehr. QD-Score 121.18 bei CMA-MAE (Lernrate 0.1) gegen 121.76 bei MAP-Elites (−0.5 %).",
    "Falle": "Stufe 3 bei 140 Schritten, CMA-ME: QD-Score 101.08 gegen 102.96 bei MAP-Elites (−1.8 %); beide belegen alle 144 Zellen und lösen 20 von 20 Läufen.",
    "Zu kleine Schrittweite": "Freies Feld, 40 Schritte, CMA-ME mit σ₀ = 0.05: QD-Score nur 97.92, gegen 128.69 mit σ₀ = 0.2 und 127.34 bei MAP-Elites.",
}

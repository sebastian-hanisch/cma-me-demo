"""CMA-ME und CMA-MAE – wann lohnt es sich, die Suchverteilung zu lernen? - Stück 6 der Konzepte-Linie "Novelty Search und Quality-Diversity"
Sebastian Hanisch - Operations Research und Machine Learning

Derselbe Hof-Roboter, dasselbe Archiv und dieselbe Qualität (Direktheit) wie in Stück 5 - aber die Kinder entstehen nicht mehr durch feste Mutation, sondern aus CMA-ES-Emittern, die die Richtung der Verbesserung lernen (CMA-ME) und
mit Schwellen je Zelle (CMA-MAE). Gemessen wird, ab welcher Dimension des Suchraums das nichts mehr bringt. Siehe README für die Einordnung in die Linie.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import cme_constants as C
import cme_evaluation as E
import cme_maze as M
from cme_presets import (apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params)
from cme_visualization import (build_archive_map, build_cases_chart, build_condition_chart, build_coverage_curve, build_paths_figure, build_qd_curve, build_sigma_chart, build_study_curve)

st.set_page_config(page_title="CMA-ME – Sebastian Hanisch", layout="wide")

PRE = E.load_precomputed()
N = C.STUDY_SEEDS


@st.cache_data(show_spinner=False)
def _pair(level, gens, seed, steps, sigma, method):
    return E.live_pair(level, gens, seed, steps, sigma, method)


@st.cache_data(show_spinner=False)
def _solvability(level):
    return E.solvability(level)


def _cm(case, method):
    return E.study_cell(PRE, case, method)


def _qd(case, method):
    return np.array(_cm(case, method)["qd"], float)


def _gain(case, method):
    """Relativer Gewinn des mittleren QD-Scores von `method` gegenüber MAP-Elites im Fall `case`."""
    return float(_qd(case, method).mean() / _qd(case, "me").mean() - 1)


def _pct(x):
    """Anteil als Prozent mit Vorzeichen, einer Nachkommastelle und Leerzeichen (0.1966 -> "+19.7 %", -0.005 -> "−0.5 %")."""
    return f"{'+' if x >= 0 else '−'}{abs(100 * x):.1f} %"


def _pstr(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _dim_label(steps):
    return f"{steps} Schritte ({2 * steps} Dimensionen)"


st.title("🧭 CMA-ME und CMA-MAE – wann lohnt es sich, die Suchverteilung zu lernen?")
st.markdown(
    """
MAP-Elites erzeugt Kinder durch **feste Mutation**: ein zufälliger Elite, ein paar Gene verrauscht. **CMA-ME** gibt jeder Teilsuche (einem **Emitter**) stattdessen ein **CMA-ES**: eine Normalverteilung, die ihre Richtung und Stärke lernt, wohin sich
Verbesserungen des Archivs ergeben, und mit der die nächsten Kinder gezogen werden. **CMA-MAE** ergänzt je Zelle eine **Schwelle**, die nur langsam (mit einer Lernrate α) zur besten Qualität aufsteigt, damit Emitter auch bei kleinen Fortschritten
weiterlernen. Das verspricht schnelleres „Ausleuchten“ des Archivs. Dieses Stück misst, **wann das stimmt**: für kurze Steuerfolgen (wenige Dimensionen) deutlich, für die 280 Dimensionen der Fahrten aus den früheren Stücken nicht.
"""
)
st.caption(
    "Stück 6 der Linie „Novelty Search und Quality-Diversity“; Labyrinth, Roboter, Archiv (Zellgröße 2) und Qualität (Direktheit) wie in [Stück 5](https://sebastianhanisch-map-elites-demo.streamlit.app/). "
    "Die Umsetzung der Emitter ist eine eigene in NumPy, keine Referenzbibliothek."
)

with st.expander("So arbeiten die Emitter", expanded=True):
    st.markdown(
        """
- **MAP-Elites (Bezug):** je Generation 100 Kinder aus gleichverteilt gezogenen Elites, 15 % der Gene mit Gauß 0.25 verrauscht.
- **Emitter:** 5 Emitter mit je 20 Kindern je Generation. Jeder hält ein **CMA-ES** (Mittelwert, Schrittweite σ, volle Kovarianzmatrix) im Raum der Steuerfolgen (Dimension = 2 × Länge). Er startet bei einem zufälligen Elite mit Schrittweite σ₀.
- **CMA-ME (Verbesserungs-Emitter):** jedes Kind wird ins Archiv eingetragen. Der Emitter lernt aus den Kindern, die eine **neue Zelle** erschlossen oder einen Elite **verbesserten** (erst die neuen Zellen, dann nach Größe der Verbesserung). Brachte eine Generation nichts,
  startet er neu bei einem zufälligen Elite.
- **CMA-MAE (Lernrate α):** je Zelle eine **Schwelle**, zu Beginn 0. Verbesserung Δ = Qualität − Schwelle; ist die Qualität über der Schwelle, ersetzt das Kind den Insassen des Emitter-Archivs und die Schwelle steigt um α · (Qualität − Schwelle). Der Emitter ordnet **alle** Kinder
  nach Δ und lernt aus der besseren Hälfte; er startet neu, wenn sein CMA-ES stagniert oder zusammengefallen ist. Mit α = 0.5 steigen die Schwellen schnell, mit 0.01 langsam.
- **Gemessen** wird in einem **Ergebnisarchiv** (je Zelle die beste Fahrt aller bewerteten Kinder), bei allen Verfahren gleich: **QD-Score** = Summe der Direktheit über die belegten Zellen, dazu Abdeckung und beste Direktheit.
- Die **Konditionszahl** der Kovarianz zeigt, ob ein Emitter etwas gelernt hat: 1 heißt kugelförmig (nichts gelernt), große Werte heißen: die Verteilung bevorzugt bestimmte Richtungen.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESET_ORDER))
for i, name in enumerate(C.PRESET_ORDER):
    with preset_cols[i]:
        st.button(name, key=f"preset_{name}", width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name] or None)

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    level = st.select_slider("Labyrinth (Stufe)", options=C.LEVELS, key="level_select", format_func=lambda v: C.LEVEL_LABELS[v], help="0 freies Feld bis 4 Schlangenlinie; je höher, desto länger der Umweg.")
    steps = st.select_slider("Länge der Steuerfolge", options=C.STEP_OPTIONS, key="steps_select", format_func=_dim_label, help="Schritte der Fahrt; die Dimension des Suchraums ist doppelt so groß (Drehung und Tempo je Schritt).")
    method = st.selectbox("CMA-Verfahren", options=C.CMA_METHODS, key="method_select", format_func=lambda v: C.METHOD_LABELS[v], help="Läuft neben MAP-Elites mit gleichem Seed, gleicher Länge und gleichem Budget.")
    sigma = st.select_slider("Anfangsschrittweite σ₀ der Emitter", options=C.SIGMAS, key="sigma_select", format_func=lambda v: f"{v:g}", help="Gilt für alle CMA-Verfahren (nicht für MAP-Elites): so breit streut ein Emitter bei seinem Start um den Elite.")
    gens = st.slider("Generationen im Live-Lauf", *bounds("gens_slider"), step=C.GENS_STEP, key="gens_slider", help=f"Je Generation {C.POP} Fahrten, also {C.POP} Evaluationen.")
    seed = st.number_input("Zufalls-Seed", min_value=bounds("seed_input")[0], max_value=bounds("seed_input")[1], step=1, key="seed_input", help="Bestimmt die Anfangspopulation und alle Zufallsentscheidungen; beide Verfahren starten gleich.")
    st.button("🎲 Neuen Lauf würfeln", on_click=randomize_seed)

level, steps, method, sigma, gens, seed = int(level), int(steps), str(method), float(sigma), int(gens), int(seed)
sync_query_params({"level_select": level, "steps_select": steps, "method_select": method, "sigma_select": sigma, "gens_slider": gens, "seed_input": seed})

grid = M.make_maze(level)
lm = E.level_metrics(level)

st.markdown("---")
st.markdown("## 🔗 Wie täuschend ist diese Stufe?")
st.caption(f"{C.LEVEL_LABELS[level]}. Die Maße stammen aus Stück 1. Hier geht es um das Archiv, nicht um das Ziel: keines der Verfahren benutzt das Ziel.")
m1 = st.columns(4)
m1[0].metric("Kürzester Weg", f"{lm['path']} Zellen", delta=f"{lm['detour']:.2f}-facher Umweg", delta_color="off")
m1[1].metric("Fitness-Distanz-Korrelation", f"{lm['fdc']:.2f}", help="1 = die Luftlinie führt zum Ziel; je kleiner, desto täuschender.")
m1[2].metric("Fallen (lokale Minima)", f"{lm['minima']}", help="Zellen, in denen die Luftlinie zum Ziel kleiner ist als in allen freien Nachbarzellen.")
m1[3].metric("Freie Zellen", f"{lm['free']}", help="So viele Zellen kann eine Endposition belegen.")
dist_ok, steps_used = _solvability(level)
st.caption(f"Lösbar? Ja: eine aus dem kürzesten Pfad abgeleitete Steuerfolge fährt ins Ziel (Abstand am Ende {dist_ok:.2f} Zellen, {steps_used} von {C.STEPS} Schritten benutzt).")
reach = steps * C.SPEED_MAX
if reach < lm["straight"] - C.SUCCESS_RADIUS:
    st.caption(f"Mit {steps} Schritten fährt der Roboter höchstens {reach:.1f} Zellen weit; die Luftlinie von Start zu Ziel ist {lm['straight']:.0f} Zellen lang (Erfolgsradius {C.SUCCESS_RADIUS:g}). Das Ziel ist hier unerreichbar, „Evaluationen bis zum Erfolg“ bleibt deshalb leer; es geht um das Archiv.")

st.markdown("---")
st.subheader("📐 Ein Lauf: MAP-Elites gegen ein CMA-Verfahren")
with st.spinner(f"Lasse beide Verfahren {gens} Generationen laufen …"):
    (me, me_q, me_tracks, _), (cm, cm_q, cm_tracks, _) = _pair(level, gens, seed, steps, sigma, method)
total = me.total
c_me, c_cm = st.columns(2)
with c_me:
    st.markdown(f"**{C.METHOD_LABELS['me']}**")
    a1 = st.columns(2)
    a1[0].metric("MAP-Elites: belegte Zellen", f"{me.coverage[-1]} von {total}")
    a1[1].metric("MAP-Elites: QD-Score", f"{me.qd_score[-1]:.2f}")
    a2 = st.columns(2)
    a2[0].metric("MAP-Elites: beste Direktheit", f"{me.best_quality[-1]:.3f}")
    a2[1].metric("MAP-Elites: Evaluationen bis zum Erfolg", C.fmt_int(me.first_solved) if me.first_solved is not None else "–")
    st.plotly_chart(build_archive_map(grid, me.archive_q), width="stretch", key=f"map_me_{level}_{steps}_{gens}_{seed}")
    st.plotly_chart(build_paths_figure(grid, me_tracks, me_q), width="stretch", key=f"paths_me_{level}_{steps}_{gens}_{seed}")
with c_cm:
    st.markdown(f"**{C.METHOD_LABELS[method]}** (σ₀ = {sigma:g}, {C.N_EMITTERS} Emitter)")
    b1 = st.columns(2)
    b1[0].metric("CMA: belegte Zellen", f"{cm.coverage[-1]} von {total}")
    b1[1].metric("CMA: QD-Score", f"{cm.qd_score[-1]:.2f}")
    b2 = st.columns(3)
    b2[0].metric("CMA: beste Direktheit", f"{cm.best_quality[-1]:.3f}")
    b2[1].metric("CMA: Evaluationen bis zum Erfolg", C.fmt_int(cm.first_solved) if cm.first_solved is not None else "–")
    b2[2].metric("CMA: Neustarts der Emitter", f"{cm.restarts}")
    st.plotly_chart(build_archive_map(grid, cm.archive_q), width="stretch", key=f"map_cm_{level}_{steps}_{gens}_{seed}_{method}_{sigma}")
    st.plotly_chart(build_paths_figure(grid, cm_tracks, cm_q), width="stretch", key=f"paths_cm_{level}_{steps}_{gens}_{seed}_{method}_{sigma}")
st.caption("Oben: die Archiv-Karte (jede Zelle zeigt die Direktheit ihres Elites; grau = leer, dunkel = Wand). Unten: die Fahrten aller Elites, Endpunkte nach Direktheit gefärbt (hell = direkter).")
col_c, col_d = st.columns(2)
with col_c:
    st.markdown("**Belegte Zellen über den Generationen**")
    st.plotly_chart(build_coverage_curve(me.coverage, cm.coverage, method, total), width="stretch", key=f"cov_{level}_{steps}_{gens}_{seed}_{method}_{sigma}")
with col_d:
    st.markdown("**QD-Score über den Generationen**")
    st.plotly_chart(build_qd_curve(me.qd_score, cm.qd_score, method), width="stretch", key=f"qd_{level}_{steps}_{gens}_{seed}_{method}_{sigma}")
gain = cm.qd_score[-1] / me.qd_score[-1] - 1 if me.qd_score[-1] > 0 else float("nan")
st.info(
    f"Nach {gens} Generationen hat das CMA-Verfahren einen QD-Score von {cm.qd_score[-1]:.2f} (MAP-Elites: {me.qd_score[-1]:.2f}, {_pct(gain)}), belegt {cm.coverage[-1]} Zellen (MAP-Elites: {me.coverage[-1]}) und hat seine Emitter {cm.restarts}-mal neu gestartet. "
    f"Das ist ein Lauf; die Studie unten vergleicht {N} Läufe."
)

st.markdown("---")
st.subheader("🔬 Wann gewinnt CMA? Die Studie über 20 Läufe")
st.markdown(
    f"Je Fall und Verfahren {N} Läufe mit unterschiedlichem Seed, jeweils {C.fmt_int(PRE['budget'])} Evaluationen ({C.STUDY_GENS} Generationen), Anfangsschrittweite σ₀ = {C.DEFAULT_SIGMA:g}. **Freies Feld** mit 10 bis 140 Schritten ändert nur die Dimension des Suchraums "
    "(20 bis 280); **Stufe 3 und 4** sind die Labyrinthe aus den früheren Stücken bei 140 Schritten. Alle Kennzahlen kommen aus dem Ergebnisarchiv der Zellgröße 2 (144 Archivzellen)."
)
st.plotly_chart(build_cases_chart(PRE), width="stretch", key="cases_chart")
head = "| Gewinn des QD-Scores gegenüber MAP-Elites | " + " | ".join(C.CASE_LABELS[c] for c in C.CASES) + " |\n|---|" + "---|" * len(C.CASES) + "\n"
rows = []
for m in C.CMA_METHODS:
    cells = []
    for case in C.CASES:
        p = E.mannwhitney_p(_qd(case, m), _qd(case, "me"))
        cells.append(f"{_pct(_gain(case, m))} ({_pstr(p)})")
    rows.append(f"| {C.METHOD_LABELS[m]} | " + " | ".join(cells) + " |")
st.markdown(head + "\n".join(rows))
study_case = st.select_slider("Fall für die Einzelheiten", options=list(C.CASES), value="d40", key="study_case_select", format_func=lambda v: C.CASE_LABELS[v])
rows = []
for m in C.METHODS:
    c = _cm(study_case, m)
    qd = _qd(study_case, m)
    solved = E.success_count(c["runs"])
    rows.append(f"| {C.METHOD_LABELS[m]} | {qd.mean():.2f} ± {qd.std(ddof=1):.2f} | {np.mean(c['coverage']):.2f} von {PRE['cases'][study_case]['total']} | {np.mean(c['best_quality']):.4f} | {np.mean(c['restarts']):.0f} | "
                f"{solved} von {N} | {np.mean(c['progress']):.2f} |")
st.markdown(f"**{C.CASE_LABELS[study_case]}**\n\n| Verfahren | QD-Score (Mittel ± Standardabweichung) | belegte Zellen | beste Direktheit | Neustarts je Lauf | Erfolg | Restweg |\n|---|---|---|---|---|---|---|\n" + "\n".join(rows))
col_e, col_f = st.columns(2)
with col_e:
    st.markdown("**QD-Score über den Evaluationen**")
    st.plotly_chart(build_study_curve(PRE, study_case), width="stretch", key=f"study_curve_{study_case}")
with col_f:
    st.markdown("**Hat die Kovarianz etwas gelernt?**")
    st.plotly_chart(build_condition_chart(PRE, study_case), width="stretch", key=f"condition_{study_case}")
low = {m: _gain("d20", m) for m in C.CMA_METHODS}
mid = {m: _gain("d80", m) for m in C.CMA_METHODS}
nat = {m: _gain("d140", m) for m in C.CMA_METHODS}
maze = [_gain(case, m) for case in ("l3", "l4") for m in C.CMA_METHODS]
restarts_me = [float(np.mean(_cm(case, "cmame")["restarts"])) for case in C.CASES]
cond = {case: _cm(case, "mae10")["log_condition"]["400"] for case in ("d10", "d140")}
st.info(
    f"**Wenige Dimensionen (20 Schritte, 40 Dimensionen):** CMA-ME {_pct(low['cmame'])}, CMA-MAE mit Lernrate 0.01 / 0.1 / 0.5 {_pct(low['mae01'])} / {_pct(low['mae10'])} / {_pct(low['mae50'])} im QD-Score gegenüber MAP-Elites; "
    f"bei 80 Schritten (160 Dimensionen) {_pct(mid['cmame'])} und {_pct(mid['mae01'])} / {_pct(mid['mae10'])} / {_pct(mid['mae50'])}; **bei 140 Schritten (280 Dimensionen, die Länge der früheren Stücke)** {_pct(nat['cmame'])} und "
    f"{_pct(nat['mae01'])} / {_pct(nat['mae10'])} / {_pct(nat['mae50'])}. Der Vorsprung schwindet mit der Dimension. Auf Stufe 3 und 4 bei 140 Schritten liegen alle CMA-Verfahren zwischen {_pct(min(maze))} und {_pct(max(maze))} gegenüber MAP-Elites, also leicht darunter. "
    f"**Was die Messung zur Ursache zeigt:** die CMA-ME-Emitter starten im Mittel {C.fmt_int(round(min(restarts_me)))} bis {C.fmt_int(round(max(restarts_me)))} Mal je Lauf neu (von {C.fmt_int(C.N_EMITTERS * (C.STUDY_GENS - 1))} Emitter-Generationen) und behalten eine kugelförmige Kovarianz (Konditionszahl 1); "
    f"die Emitter von CMA-MAE mit Lernrate 0.1 lernen etwas mehr, aber wenig und mit der Dimension weniger (Zehnerlogarithmus der Konditionszahl {cond['d10']:.2f} bei 20 Dimensionen, {cond['d140']:.2f} bei 280). Das passt zu der Erklärung, "
    "dass die Kovarianz mit nur 20 Kindern je Emitter und Generation in vielen Dimensionen nicht zu lernen ist, beweist sie aber nicht."
)
st.markdown("---")
st.subheader("🔬 Wie wichtig ist die Anfangsschrittweite σ₀?")
st.markdown(f"Freies Feld mit 40 und 140 Schritten, σ₀ = 0.05 / 0.1 / 0.2 / 0.4, je {N} Läufe. Gestrichelte Linien: MAP-Elites (das keine Schrittweite kennt).")
st.plotly_chart(build_sigma_chart(PRE), width="stretch", key="sigma_chart")
s40 = {(m, sg): float(np.mean(E.sigma_cell(PRE, "s40", m, sg)["qd"])) for m in C.SIGMA_METHODS for sg in C.SIGMAS}
best40 = {m: max(C.SIGMAS, key=lambda sg: s40[(m, sg)]) for m in C.SIGMA_METHODS}
st.info(
    f"Bei 40 Schritten liegt der beste Wert für CMA-ME bei σ₀ = {best40['cmame']:g} (QD-Score {s40[('cmame', best40['cmame'])]:.2f}) und für CMA-MAE mit Lernrate 0.1 bei σ₀ = {best40['mae10']:g} ({s40[('mae10', best40['mae10'])]:.2f}); "
    f"mit σ₀ = {C.SIGMAS[0]:g} fallen sie auf {s40[('cmame', C.SIGMAS[0])]:.2f} und {s40[('mae10', C.SIGMAS[0])]:.2f}, mit σ₀ = {C.SIGMAS[-1]:g} auf {s40[('cmame', C.SIGMAS[-1])]:.2f} und {s40[('mae10', C.SIGMAS[-1])]:.2f} "
    f"(MAP-Elites: {_qd('d40', 'me').mean():.2f}). CMA-Verfahren brauchen also eine passende Anfangsschrittweite; die feste Mutation von MAP-Elites braucht diese Wahl nicht."
)

st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Suchverteilung lässt sich lernen** | Mit 280 Dimensionen und 20 Kindern je Emitter und Generation lernt die Kovarianz in 400 Generationen nichts Messbares: kein Vorsprung gegenüber MAP-Elites. Mehr Kinder je Emitter oder ein längeres Budget wurden nicht gemessen. | kein Folgestück |
| **Die Qualität gibt eine Richtung vor** | Die Direktheit ist glatt; bei flachen oder sprunghaften Qualitäten verhalten sich CMA-ME (verliert die Richtung) und CMA-MAE (Schwellen als Ersatz) anders. Nicht gemessen. | kein Folgestück |
| **Ein festes Raster genügt** | Mit hochdimensionalem Verhalten wachsen die Zellen exponentiell. | **CVT-MAP-Elites** (Folgestück) |
| **Eigene Umsetzung = Referenz** | Die Emitter sind eine eigene NumPy-Umsetzung nach den Veröffentlichungen, kein Test gegen die Referenzbibliothek des Verfahrens; Details (Neustart-Regeln, Gewichte) können abweichen. | kein Folgestück |
| **Eine Suche mit fester Steuerfolge** | Stufe 4 hängt am Suchraum (Stück 3): auch ein besseres Archiv-Verfahren löst das Ziel nicht. | **Go-Explore** und **POET** (Folgestücke) |
"""
)
st.caption(
    "Verwandt im Portfolio: [deception-maze-demo](https://sebastianhanisch-deception-maze-demo.streamlit.app/) (Stück 1), [novelty-search-demo](https://sebastianhanisch-novelty-search-demo.streamlit.app/) (Stück 2), "
    "[behavior-space-demo](https://sebastianhanisch-behavior-space-demo.streamlit.app/) (Stück 3), [novelty-objective-demo](https://sebastianhanisch-novelty-objective-demo.streamlit.app/) (Stück 4), "
    "[map-elites-demo](https://sebastianhanisch-map-elites-demo.streamlit.app/) (Stück 5), [cma-es-demo](https://sebastianhanisch-cma-es-demo.streamlit.app/) (das CMA-ES allein)."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Qualität und Archiv** wie in Stück 5: Direktheit $q(g) = \lVert p_T - p_0\rVert / \sum_t \lVert p_t - p_{t-1}\rVert \in [0, 1]$; Zelle $z(g) = (\lfloor x_T/2\rfloor, \lfloor y_T/2\rfloor)$; Ergebnisarchiv $\mathcal{A}$: je Zelle die beste Fahrt (ein Kind
ersetzt den Elite, wenn es um mehr als $10^{-12}$ besser ist). QD-Score $=\sum_{z\in\mathcal{A}} q(g^\ast_z)$.

**CMA-ES-Emitter** (Hansen 2016): Stichprobe $x_k = m + \sigma\, B D z_k$, $z_k \sim \mathcal{N}(0, I)$, $k = 1,\dots,\lambda$ ($\lambda = 20$), Dimension $n = 2\cdot\text{Schritte}$. Mit den Rängen $1,\dots,\mu$ der berücksichtigten Kinder und Gewichten
$w_i \propto \ln(\mu + \tfrac12) - \ln i$: $m \leftarrow m + \sigma\sum_i w_i y_i$, $y_i = B D z_{(i)}$; Pfade $p_\sigma, p_c$, Kovarianz $C \leftarrow (1-c_1-c_\mu)\,C + c_1 p_c p_c^\top + c_\mu \sum_i w_i y_i y_i^\top$, Schrittweite
$\sigma \leftarrow \sigma \exp\!\big(\tfrac{c_\sigma}{d_\sigma}(\lVert p_\sigma\rVert / \chi_n - 1)\big)$ mit den Konstanten des Tutorials.

**CMA-ME:** berücksichtigt werden nur Kinder, die das Ergebnisarchiv ändern, geordnet nach (neue Zelle vor Verbesserung, dann absteigend nach Qualität bzw. Verbesserung $q - q_{\text{Elite}}$); keine Änderung in einer Generation: Neustart bei einem zufälligen Elite.

**CMA-MAE:** Schwellen $t_z$ (Start 0). Für jedes Kind $\Delta = q - t_z$; gilt $q > t_z$, ersetzt es den Insassen und $t_z \leftarrow (1-\alpha)\,t_z + \alpha\, q$. Die $\mu = \lambda/2$ Kinder mit größtem $\Delta$ aktualisieren das CMA-ES; Neustart bei Konvergenz
(Schrittweite mal größte Achse unter $10^{-4}$, Konditionszahl über $10^{12}$ oder 25 Generationen ohne besseren besten Wert) bei einem zufälligen Insassen.

Implementiert in `cme_cmaes.py` (CMA-ES-Emitter) und `cme_search.py` (Archiv, Verfahren, Kennzahlen).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html))."
)

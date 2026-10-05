"""Plotly-Abbildungen der CMA-ME-Demo: Archiv-Karte (Qualität je Zelle), Bündel der Elite-Fahrten, Verläufe von Abdeckung und QD-Score, QD-Score je Fall und Verfahren (Dimension und Labyrinth), QD-Score über den Evaluationen,
Konditionszahl der Emitter-Kovarianz, Sweep der Anfangsschrittweite. Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen. Das Raster `grid[i, j]` wird mit i als x-Achse gezeichnet."""

import numpy as np
import plotly.graph_objects as go

import cme_constants as C
import cme_evaluation as E

WALL_COLOR = "#3a3f47"
EMPTY_COLOR = "#e4e7ec"
START_COLOR, GOAL_COLOR, PATH_COLOR = "#54a24b", "#e45756", "#4c78a8"
ME_COLOR = "#2ca25f"
METHOD_COLORS = {"me": "#2ca25f", "cmame": "#e45756", "mae01": "#9ecae1", "mae10": "#3182bd", "mae50": "#08519c"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height, legend_y=-0.25, top=10):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=top, b=10), legend=dict(orientation="h", y=legend_y), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _arena_axes(fig):
    """Beide Achsen 0 bis N mit gleichem Maßstab (quadratische Zellen), Zeichenfläche passt sich der kleineren Seite an (constrain=domain)."""
    fig.update_xaxes(range=[0, C.N], showgrid=False, zeroline=False, title_text="x (Zellen)", constrain="domain")
    fig.update_yaxes(range=[0, C.N], showgrid=False, zeroline=False, title_text="y (Zellen)", scaleanchor="x", scaleratio=1, constrain="domain")


def _walls(fig, grid):
    z = np.where(grid, 1.0, np.nan)
    fig.add_trace(go.Heatmap(z=z.T, x0=0.5, dx=1, y0=0.5, dy=1, zmin=0, zmax=1, colorscale=[[0, WALL_COLOR], [1, WALL_COLOR]], showscale=False, hoverinfo="skip", xgap=0, ygap=0))


def _markers(fig):
    fig.add_trace(go.Scatter(x=[C.START[0]], y=[C.START[1]], mode="markers", marker=dict(color=START_COLOR, size=12, symbol="square", line=dict(color="white", width=1.5)), name="Start"))
    fig.add_trace(go.Scatter(x=[C.GOAL[0]], y=[C.GOAL[1]], mode="markers", marker=dict(color=GOAL_COLOR, size=15, symbol="star", line=dict(color="white", width=1.5)), name="Ziel"))
    fig.add_shape(type="circle", xref="x", yref="y", x0=C.GOAL[0] - C.SUCCESS_RADIUS, x1=C.GOAL[0] + C.SUCCESS_RADIUS, y0=C.GOAL[1] - C.SUCCESS_RADIUS, y1=C.GOAL[1] + C.SUCCESS_RADIUS,
                  line=dict(color=GOAL_COLOR, dash="dot", width=1.5))


def build_archive_map(grid, quality):
    """Archiv-Karte: jede Zelle der Kantenlänge 2 zeigt die Direktheit ihres Elites (hell = direkter); leere Zellen sind grau, Wände dunkel (in der Auflösung des Labyrinths darübergelegt)."""
    z = np.where(quality >= 0, quality, np.nan)
    fig = go.Figure(go.Heatmap(z=z.T, x0=C.CELL / 2, dx=C.CELL, y0=C.CELL / 2, dy=C.CELL, zmin=0, zmax=1, colorscale="Viridis", xgap=1, ygap=1, hoverinfo="skip",
                               colorbar=dict(title=dict(text="Direktheit des Elites", side="top"), orientation="h", thickness=10, len=0.9, y=-0.2, yanchor="top")))
    _walls(fig, grid)
    _markers(fig)
    _arena_axes(fig)
    fig.update_layout(plot_bgcolor=EMPTY_COLOR)
    return _base(fig, 470, legend_y=-0.55, top=0)


def build_paths_figure(grid, tracks, quality):
    """Bündel der Elite-Fahrten (eine dünne Linie je belegter Zelle); Endpunkte nach Direktheit gefärbt."""
    xs, ys = [], []
    for t in tracks:
        xs += list(t[:, 0]) + [None]
        ys += list(t[:, 1]) + [None]
    fig = go.Figure()
    _walls(fig, grid)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=PATH_COLOR, width=1), opacity=0.3, name="Fahrten der Elites", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=tracks[:, -1, 0], y=tracks[:, -1, 1], mode="markers", name="Endpunkte der Elites", hoverinfo="skip",
                             marker=dict(color=quality, colorscale="Viridis", cmin=0, cmax=1, size=5, line=dict(color="white", width=0.4))))
    _markers(fig)
    _arena_axes(fig)
    return _base(fig, 470, legend_y=-0.3, top=0)


def _pair_curve(me_y, other_y, method, y_title, tickformat=None, y_range=None):
    gens = list(range(1, len(me_y) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gens, y=me_y, mode="lines", line=dict(color=METHOD_COLORS["me"], width=2.5), name=C.METHOD_LABELS["me"]))
    fig.add_trace(go.Scatter(x=gens, y=other_y, mode="lines", line=dict(color=METHOD_COLORS[method], width=2.5), name=C.METHOD_LABELS[method]))
    fig.update_xaxes(title_text="Generation")
    fig.update_yaxes(title_text=y_title, rangemode="tozero", tickformat=tickformat, range=y_range)
    return _base(fig, 320, legend_y=-0.35)


def build_coverage_curve(me_cov, other_cov, method, total):
    """Anteil der Archivzellen mit freier Fläche, die belegt sind, über den Generationen: MAP-Elites (Bezug) gegen ein CMA-Verfahren."""
    return _pair_curve([c / total for c in me_cov], [c / total for c in other_cov], method, "belegte Zellen (Anteil)", tickformat=".0%", y_range=[0, 1.03])


def build_qd_curve(me_qd, other_qd, method):
    """QD-Score (Summe der Direktheit über die belegten Zellen) über den Generationen: MAP-Elites (Bezug) gegen ein CMA-Verfahren."""
    return _pair_curve(me_qd, other_qd, method, "QD-Score")


def build_cases_chart(pre):
    """Mittlerer QD-Score je Fall (freies Feld mit 10 bis 140 Schritten, Stufe 3 und 4) und Verfahren; Fehlerbalken = Standardabweichung über die 20 Läufe."""
    fig = go.Figure()
    cases = list(C.CASES)
    for m in C.METHODS:
        means, sds = [], []
        for case in cases:
            v = np.array(E.study_cell(pre, case, m)["qd"], float)
            means.append(float(v.mean()))
            sds.append(float(v.std(ddof=1)))
        fig.add_trace(go.Bar(x=[C.CASE_LABELS[c] for c in cases], y=means, name=C.METHOD_LABELS[m], marker_color=METHOD_COLORS[m], error_y=dict(type="data", array=sds, color="#555", thickness=1)))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="QD-Score nach 400 Generationen (Mittel ± Standardabweichung)", rangemode="tozero")
    fig.update_xaxes(tickangle=-20)
    return _base(fig, 420, legend_y=-0.45)


def build_study_curve(pre, case):
    """Mittlerer QD-Score über den Evaluationen im gewählten Fall, ein Linienzug je Verfahren."""
    fig = go.Figure()
    for m in C.METHODS:
        curve = E.study_cell(pre, case, m)["mean_qd"]
        fig.add_trace(go.Scatter(x=[(i + 1) * C.POP for i in range(len(curve))], y=curve, mode="lines", line=dict(color=METHOD_COLORS[m], width=2.5), name=C.METHOD_LABELS[m]))
    fig.update_xaxes(title_text="Evaluationen")
    fig.update_yaxes(title_text="mittlerer QD-Score", rangemode="tozero")
    return _base(fig, 340, legend_y=-0.45)


def build_condition_chart(pre, case):
    """Mittlerer Zehnerlogarithmus der Konditionszahl der Emitter-Kovarianz (über Emitter und Läufe) zu den Meilensteinen, je CMA-Verfahren; 0 heißt: kugelförmig, die Verteilung hat nichts gelernt."""
    fig = go.Figure()
    for m in C.CMA_METHODS:
        cond = E.study_cell(pre, case, m)["log_condition"]
        fig.add_trace(go.Scatter(x=[int(g) for g in cond], y=list(cond.values()), mode="lines+markers", line=dict(color=METHOD_COLORS[m], width=2.5), name=C.METHOD_LABELS[m]))
    fig.update_xaxes(title_text="Generation", type="log", tickvals=list(C.MILESTONES))
    fig.update_yaxes(title_text="log10 der Konditionszahl der Kovarianz", rangemode="tozero")
    return _base(fig, 340, legend_y=-0.45)


def build_sigma_chart(pre):
    """Mittlerer QD-Score nach 400 Generationen über der Anfangsschrittweite σ₀, für CMA-ME und CMA-MAE (Lernrate 0.1) im freien Feld bei 40 und 140 Schritten; gestrichelt: MAP-Elites mit derselben Länge."""
    fig = go.Figure()
    for case, dash in (("s40", "solid"), ("s140", "dot")):
        steps = C.SIGMA_CASES[case][1]
        for m in C.SIGMA_METHODS:
            ys = [float(np.mean(E.sigma_cell(pre, case, m, sg)["qd"])) for sg in C.SIGMAS]
            fig.add_trace(go.Scatter(x=[f"{sg:g}" for sg in C.SIGMAS], y=ys, mode="lines+markers", line=dict(color=METHOD_COLORS[m], width=2.5, dash=dash), name=f"{C.METHOD_LABELS[m]}, {steps} Schritte"))
        me = float(np.mean(E.study_cell(pre, f"d{steps}", "me")["qd"]))
        fig.add_hline(y=me, line=dict(color=METHOD_COLORS["me"], dash="dash", width=1.5), annotation_text=f"MAP-Elites, {steps} Schritte", annotation_position="right")
    fig.update_xaxes(title_text="Anfangsschrittweite σ₀", type="category")
    fig.update_yaxes(title_text="QD-Score nach 400 Generationen (Mittel)", rangemode="tozero")
    return _base(fig, 380, legend_y=-0.5)

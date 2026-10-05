"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).
Alle Regler sind immer sichtbar und wirken immer: MAP-Elites läuft im Live-Lauf als Bezug neben dem gewählten CMA-Verfahren mit derselben Genomlänge; die Anfangsschrittweite gilt für alle CMA-Verfahren, die Lernrate steckt in der Wahl des Verfahrens
(CMA-MAE mit 0.01 / 0.1 / 0.5), es gibt also keinen ausblendbaren Regler und kein KEPT-Muster."""

import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import cme_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _method(value):
    """Permalink-Wert des CMA-Verfahrens: nur CMA-ME und die drei CMA-MAE, sonst ValueError (der Wert wird dann ignoriert)."""
    if not isinstance(value, str) or value not in C.CMA_METHODS:
        raise ValueError(f"unbekanntes Verfahren: {value!r}")
    return value


SETTING_SPECS = {
    "level_select": SettingSpec("level", int, C.DEFAULT_LEVEL, C.LEVELS[0], C.LEVELS[-1]),
    "gens_slider": SettingSpec("gens", int, C.DEFAULT_GENS, C.GENS_MIN, C.GENS_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "steps_select": SettingSpec("steps", int, C.DEFAULT_STEPS, C.STEP_OPTIONS[0], C.STEP_OPTIONS[-1]),
    "sigma_select": SettingSpec("sigma", float, C.DEFAULT_SIGMA, C.SIGMAS[0], C.SIGMAS[-1]),
    "method_select": SettingSpec("method", _method, C.DEFAULT_METHOD),
}
PRESET_KEYS = {"level": "level_select", "gens": "gens_slider", "seed": "seed_input", "steps": "steps_select", "sigma": "sigma_select", "method": "method_select"}
OPTIONS = {"level_select": C.LEVELS, "steps_select": C.STEP_OPTIONS, "sigma_select": C.SIGMAS}       # Regler, die nur feste Werte kennen


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def snap_to_option(state_key, value):
    """Regler mit festen Werten: ein Permalink-Wert dazwischen rastet auf den nächsten Wert ein (bei Gleichstand auf den kleineren)."""
    return min(OPTIONS[state_key], key=lambda o: (abs(o - value), o))


def snap_gens(value):
    """Die Generationen rasten auf das nächste Vielfache der Schrittweite (10, von 10 an) innerhalb der Grenzen ein."""
    snapped = C.GENS_MIN + round((value - C.GENS_MIN) / C.GENS_STEP) * C.GENS_STEP
    return int(min(C.GENS_MAX, max(C.GENS_MIN, snapped)))


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if value != value:                     # NaN
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                if state_key in OPTIONS:
                    value = snap_to_option(state_key, value)
                if state_key == "gens_slider":
                    value = snap_gens(value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)

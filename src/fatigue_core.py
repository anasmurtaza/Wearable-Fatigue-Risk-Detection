# -*- coding: utf-8 -*-
"""
fatigue_core.py - shared definitions for the Wearable Fatigue Risk project
=========================================================================
Used by the data generator, the training script, the notebook and the GUI app.

It contains
  * the list of wearable signals (heart rate, movement, sleep, work schedule),
  * feature engineering (personal-baseline features + a circadian-rhythm term),
  * an explainable RULE-BASED fatigue score (the baseline the ML model must beat),
  * plain-language advice for each risk level.

IMPORTANT: this is a research / decision-support prototype. It is NOT a medical
device and must not be the only safety barrier in a workplace or in a vehicle.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
DATA_FILE = DATA_DIR / "wearable_fatigue_data.csv"
MODEL_FILE = MODEL_DIR / "fatigue_model.joblib"

# ---------------------------------------------------------------- labels
LABELS = ["Low", "Moderate", "High"]          # fatigue risk classes (ordered)

# Raw signals a wearable (+ shift roster) can provide for one monitoring window
RAW_COLUMNS = [
    # sleep (last night / last week)
    "sleep_hours",         # hours slept last night
    "sleep_efficiency",    # % of time in bed actually asleep
    "awakenings",          # number of awakenings last night
    "sleep_debt_7d",       # hours of missing sleep over the last 7 nights (vs 7.5 h)
    # work schedule
    "hours_awake",         # hours since the person woke up
    "shift_hours",         # hours already worked / driven in this shift
    "hour_of_day",         # clock hour 0-23
    # heart rate
    "hr_mean",             # mean heart rate in the window (bpm)
    "hr_resting",          # the person's resting heart-rate baseline (bpm)
    "hrv_rmssd",           # heart-rate variability RMSSD in the window (ms)
    "hrv_baseline",        # the person's normal RMSSD (ms)
    # movement (accelerometer)
    "activity_level",      # mean activity counts per minute
    "idle_min_per_hr",     # minutes per hour with (almost) no movement
    "movement_var",        # variability of movement intensity (0-1)
]

# Features actually given to the machine-learning models
MODEL_FEATURES = [
    "sleep_hours", "sleep_efficiency", "awakenings", "sleep_debt_7d",
    "hours_awake", "shift_hours", "circadian_risk",
    "hr_mean", "hr_delta", "hrv_ratio",
    "activity_level", "idle_min_per_hr", "movement_var",
]

# Nice names for plots and the app
PRETTY = {
    "sleep_hours": "Sleep last night (h)", "sleep_efficiency": "Sleep efficiency (%)",
    "awakenings": "Awakenings", "sleep_debt_7d": "7-day sleep debt (h)",
    "hours_awake": "Hours awake", "shift_hours": "Hours on shift",
    "circadian_risk": "Circadian low-point risk", "hr_mean": "Heart rate (bpm)",
    "hr_delta": "HR above resting (bpm)", "hrv_ratio": "HRV vs own baseline",
    "activity_level": "Activity level", "idle_min_per_hr": "Idle min / hour",
    "movement_var": "Movement variability",
}


# ---------------------------------------------------------------- features
def circadian_risk(hour):
    """Body-clock sleepiness curve (0 = alert ... 0.8 = deepest low).

    People are sleepiest around 03:00-05:00 (main low) and, less strongly, around
    14:00-16:00 (post-lunch dip). Modelled as two smooth bumps on a 24 h clock.
    """
    hour = np.asarray(hour, dtype=float)
    night = np.minimum(np.abs(hour - 4.0), 24 - np.abs(hour - 4.0))
    afternoon = np.abs(hour - 15.0)
    return 0.8 * np.exp(-(night ** 2) / (2 * 2.5 ** 2)) + 0.3 * np.exp(-(afternoon ** 2) / (2 * 1.5 ** 2))


def engineer(df):
    """Add the derived (personal-baseline) features to a DataFrame of raw columns."""
    out = df.copy()
    out["circadian_risk"] = circadian_risk(out["hour_of_day"])
    out["hr_delta"] = out["hr_mean"] - out["hr_resting"]
    out["hrv_ratio"] = out["hrv_rmssd"] / out["hrv_baseline"]
    return out


# ---------------------------------------------------------------- rule-based score
def rule_score(row):
    """Explainable expert-style score.  Returns (total_points, reasons).

    `reasons` is a list of (points, text) for every factor that adds risk, so the
    GUI can tell the user WHY a risk level was given.
    """
    r = dict(row)                                   # plain dict = fast, works for a Series too
    r["circadian_risk"] = float(circadian_risk(r["hour_of_day"]))
    r["hrv_ratio"] = r["hrv_rmssd"] / r["hrv_baseline"]
    reasons = []

    def add(points, text):
        if points:
            reasons.append((points, text))

    # --- sleep
    s = r["sleep_hours"]
    add(3 if s < 5 else 2 if s < 6 else 1 if s < 7 else 0, "Short sleep last night (%.1f h)" % s)
    d = r["sleep_debt_7d"]
    add(2 if d > 10 else 1 if d > 5 else 0, "Accumulated sleep debt (%.1f h this week)" % d)
    add(1 if r["sleep_efficiency"] < 75 else 0, "Poor sleep quality (%.0f %% efficiency)" % r["sleep_efficiency"])
    add(1 if r["awakenings"] >= 5 else 0, "Broken sleep (%d awakenings)" % r["awakenings"])
    # --- time awake / time on task / body clock
    a = r["hours_awake"]
    add(3 if a > 20 else 2 if a > 16 else 1 if a > 14 else 0, "Awake for a long time (%.0f h)" % a)
    w = r["shift_hours"]
    add(2 if w > 10 else 1 if w > 8 else 0, "Long time on shift / at the wheel (%.1f h)" % w)
    c = r["circadian_risk"]
    add(2 if c > 0.5 else 1 if c > 0.25 else 0, "Body-clock low point (around %02d:00)" % r["hour_of_day"])
    # --- heart rate variability
    v = r["hrv_ratio"]
    add(2 if v < 0.70 else 1 if v < 0.85 else 0, "HRV is %.0f %% below the person's normal" % ((1 - v) * 100))
    # --- movement
    i = r["idle_min_per_hr"]
    add(2 if i > 25 else 1 if i > 15 else 0, "Very little movement (%.0f idle min / h)" % i)
    add(1 if r["activity_level"] < 120 else 0, "Low activity level")

    reasons.sort(key=lambda item: -item[0])
    return sum(p for p, _ in reasons), reasons


def rule_points(df):
    """Rule points for every row of a DataFrame (used to score the whole dataset)."""
    return np.array([rule_score(row)[0] for row in df.to_dict("records")])


def rule_level(points):
    """Convert the rule points into a risk class."""
    return "Low" if points <= 4 else "Moderate" if points <= 8 else "High"


ADVICE = {
    "Low": "No action needed. Keep normal breaks and hydration.",
    "Moderate": ("Take a 10-15 minute break, move around, drink water or have a short "
                 "caffeine break. Re-check in about an hour; avoid safety-critical tasks "
                 "that can wait."),
    "High": ("Stop safety-critical work or driving now. Rest or nap (15-20 min) in a safe "
             "place, tell a supervisor, and arrange a replacement or a safe ride. Do not "
             "rely on caffeine or open windows."),
}

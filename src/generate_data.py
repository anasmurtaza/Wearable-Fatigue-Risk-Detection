# -*- coding: utf-8 -*-
"""
generate_data.py - create a SYNTHETIC wearable-style dataset
============================================================
No real wearable recordings are available for this project, so the data is
simulated. Each row is one monitoring window of one person (like a 30-60 min
summary from a smart-watch / smart-band plus the shift roster).

How the simulation works (so the data is realistic and explainable)
  1. Every person gets personal baselines (resting heart rate, normal HRV),
     a usual sleep length, a shift type (day / evening / night) and a small
     personal sensitivity to fatigue.
  2. For every window the sleep and schedule signals are drawn
     (night-shift workers sleep less and are awake longer).
  3. A hidden "true fatigue" score is calculated from sleep, sleep debt, hours
     awake, time on shift and the body-clock low points, plus random noise.
  4. The body signals REACT to that hidden fatigue (HRV falls, heart rate
     creeps up, movement drops, idle time grows) - with a lot of noise, because
     real sensors are noisy.
  5. The label Low / Moderate / High is cut from the hidden score. The model
     never sees the hidden score, only the wearable signals.

Run:  python generate_data.py          (writes data/wearable_fatigue_data.csv)
"""

import numpy as np
import pandas as pd

import fatigue_core as core

# Cut points of the hidden fatigue score -> Low / Moderate / High
LOW_MAX = 1.70
MODERATE_MAX = 2.70


def generate(n_subjects=150, windows_per_subject=40, seed=42):
    """Return a DataFrame with one row per monitoring window."""
    rng = np.random.default_rng(seed)
    rows = []

    for subject in range(1, n_subjects + 1):
        # ---- personal baselines
        shift_type = rng.choice(["day", "evening", "night"], p=[0.5, 0.2, 0.3])
        rest_hr = float(np.clip(rng.normal(64, 6), 50, 85))
        hrv_base = float(np.clip(rng.lognormal(np.log(50), 0.25), 22, 95))
        usual_sleep = rng.normal(6.9, 0.7) - (0.9 if shift_type == "night" else 0.0)
        sensitivity = rng.normal(0, 0.25)            # some people tire faster

        for _ in range(windows_per_subject):
            # ---- sleep signals
            sleep_hours = float(np.clip(rng.normal(usual_sleep, 1.1), 2.5, 10.0))
            efficiency = float(np.clip(rng.normal(87, 6), 60, 99))
            awakenings = int(rng.poisson(max(0.3, (96 - efficiency) / 3.5)))
            last_week = np.clip(rng.normal(usual_sleep, 1.0, 6), 3, 10)
            nights = np.append(last_week, sleep_hours)
            sleep_debt = float(np.sum(np.maximum(0, 7.5 - nights)))

            # ---- schedule signals
            shift_hours = float(rng.uniform(0.5, 12.0))
            start = {"day": rng.uniform(6, 9), "evening": rng.uniform(14, 17),
                     "night": rng.uniform(21, 24)}[shift_type]
            hour = int((start + shift_hours) % 24)
            prep = rng.uniform(1, 3) if shift_type != "night" else rng.uniform(3, 10)
            hours_awake = float(shift_hours + prep)

            # ---- hidden "true fatigue" (never given to the model)
            circ = float(core.circadian_risk(hour))
            latent = (1.3 * max(0, 7.5 - sleep_hours) / 3
                      + 0.5 * max(0, 85 - efficiency) / 15
                      + 0.35 * min(awakenings, 8) / 8
                      + 0.9 * min(sleep_debt, 20) / 12
                      + 1.1 * max(0, hours_awake - 14) / 8
                      + 0.5 * max(0, shift_hours - 8) / 4
                      + 1.2 * circ
                      + sensitivity + rng.normal(0, 0.25))

            # ---- body signals react to fatigue (noisy)
            activity = float(np.clip(260 * np.exp(-0.18 * latent) * rng.lognormal(0, 0.25), 20, 700))
            idle = float(np.clip(8 + 3.5 * latent + rng.normal(0, 4), 0, 55))
            move_var = float(np.clip(0.55 - 0.05 * latent + rng.normal(0, 0.08), 0.1, 1.0))
            hr_mean = float(rest_hr + 5 + activity / 60 + 1.2 * latent + rng.normal(0, 3))
            hrv = float(np.clip(hrv_base * np.exp(-0.07 * latent + rng.normal(0, 0.15)), 8, 150))

            label = (core.LABELS[0] if latent < LOW_MAX
                     else core.LABELS[1] if latent < MODERATE_MAX else core.LABELS[2])

            rows.append({
                "subject_id": subject, "shift_type": shift_type,
                "sleep_hours": round(sleep_hours, 2), "sleep_efficiency": round(efficiency, 1),
                "awakenings": awakenings, "sleep_debt_7d": round(sleep_debt, 2),
                "hours_awake": round(hours_awake, 2), "shift_hours": round(shift_hours, 2),
                "hour_of_day": hour, "hr_mean": round(hr_mean, 1), "hr_resting": round(rest_hr, 1),
                "hrv_rmssd": round(hrv, 1), "hrv_baseline": round(hrv_base, 1),
                "activity_level": round(activity, 1), "idle_min_per_hr": round(idle, 1),
                "movement_var": round(move_var, 3), "fatigue_risk": label,
            })
    return pd.DataFrame(rows)


def main():
    core.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(core.DATA_FILE, index=False)
    print("Saved %d rows (%d people) to %s" % (len(df), df["subject_id"].nunique(), core.DATA_FILE))
    print(df["fatigue_risk"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
fatigue_app.py - Wearable Fatigue Risk Checker (desktop GUI)
============================================================
Move the sliders (or load an example person) and the app shows, live,

  * the predicted FATIGUE RISK  (Low / Moderate / High) and its probabilities,
  * the risk factors found in the data (why the risk is raised),
  * a second opinion from the explainable rule-based score,
  * what a supervisor / driver should do.

Inputs are the kind of values a smart-watch / smart-band and a shift roster provide:
sleep, hours awake and on shift, heart rate and HRV, and movement.

Run:  python fatigue_app.py          (needs numpy, pandas, scikit-learn, joblib)

NOTE: research prototype trained on SIMULATED data. It is NOT a medical device and
must not be the only safety barrier in a workplace or in a vehicle.
"""

import json
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

import joblib
import pandas as pd
import sklearn

import fatigue_core as core

# --------------------------------------------------------------------- colours
BG = "#f4f6f8"
HEADER_BG = "#1e293b"
RISK_COLOURS = {"Low": "#16a34a", "Moderate": "#d97706", "High": "#dc2626"}
RISK_SOFT = {"Low": "#dcfce7", "Moderate": "#fef3c7", "High": "#fee2e2"}

# ------------------------------------------------------------ the 14 input sliders
# key, label, minimum, maximum, step, default, display format
GROUPS = [
    ("Sleep", [
        ("sleep_hours", "Sleep last night", 2, 10, 0.1, 7.5, "%.1f h"),
        ("sleep_efficiency", "Sleep efficiency", 60, 99, 1, 88, "%.0f %%"),
        ("awakenings", "Awakenings at night", 0, 15, 1, 1, "%d"),
        ("sleep_debt_7d", "7-day sleep debt", 0, 25, 0.5, 2, "%.1f h"),
    ]),
    ("Work schedule", [
        ("hours_awake", "Hours awake", 1, 24, 0.5, 8, "%.1f h"),
        ("shift_hours", "Hours on shift / driving", 0, 14, 0.5, 4, "%.1f h"),
        ("hour_of_day", "Time of day", 0, 23, 1, 10, "%02d:00"),
    ]),
    ("Heart rate (wearable)", [
        ("hr_mean", "Heart rate now", 45, 120, 1, 72, "%.0f bpm"),
        ("hr_resting", "Resting heart rate (baseline)", 45, 90, 1, 62, "%.0f bpm"),
        ("hrv_rmssd", "HRV (RMSSD) now", 8, 120, 1, 52, "%.0f ms"),
        ("hrv_baseline", "HRV normal for this person", 20, 100, 1, 52, "%.0f ms"),
    ]),
    ("Movement (wearable)", [
        ("activity_level", "Activity level (counts/min)", 20, 600, 5, 230, "%.0f"),
        ("idle_min_per_hr", "Idle minutes per hour", 0, 55, 1, 8, "%.0f min"),
        ("movement_var", "Movement variability", 0.1, 1.0, 0.01, 0.55, "%.2f"),
    ]),
]

# ------------------------------------------------------------ example people
PRESETS = {
    "Rested day worker": {},                                  # = the defaults above
    "Tired night-shift worker": dict(
        sleep_hours=5.8, sleep_efficiency=82, awakenings=3, sleep_debt_7d=6, hours_awake=11,
        shift_hours=6, hour_of_day=1, hr_mean=69, hr_resting=63, hrv_rmssd=42, hrv_baseline=52,
        activity_level=190, idle_min_per_hr=15, movement_var=0.48),
    "Exhausted long-haul driver": dict(
        sleep_hours=4.0, sleep_efficiency=70, awakenings=6, sleep_debt_7d=14, hours_awake=20,
        shift_hours=10.5, hour_of_day=4, hr_mean=68, hr_resting=63, hrv_rmssd=30,
        hrv_baseline=52, activity_level=90, idle_min_per_hr=30, movement_var=0.35),
}


# ===================================================================== model loading
def load_model():
    """Load the saved model; if that is impossible, train a new one from the CSV.

    A saved scikit-learn model can break when the scikit-learn version on another
    computer is different. In that case the app quietly re-trains (takes seconds).
    Returns (model, short text describing the model).
    """
    try:
        saved = joblib.load(core.MODEL_FILE)
        if saved.get("sklearn") != sklearn.__version__:
            raise RuntimeError("scikit-learn version changed")
        model, name = saved["model"], saved["name"]
        try:
            info = json.loads((core.MODEL_DIR / "model_info.json").read_text(encoding="utf-8"))
            note = "%s | test accuracy %.0f %%, High-risk recall %.0f %%" % (
                name, info["test_scores"]["accuracy"] * 100, info["test_scores"]["high_recall"] * 100)
        except (OSError, KeyError, ValueError):
            note = name
        return model, note
    except Exception:                                          # missing / incompatible file
        import generate_data
        import train_model
        if not core.DATA_FILE.exists():
            core.DATA_DIR.mkdir(parents=True, exist_ok=True)
            generate_data.generate().to_csv(core.DATA_FILE, index=False)
        df = train_model.load_data()
        return train_model.fit_final_model(df, "Logistic Regression"), "Logistic Regression (re-trained now)"


def predict(model, values):
    """Return {'Low': p, 'Moderate': p, 'High': p} for one person."""
    frame = core.engineer(pd.DataFrame([values]))
    probs = model.predict_proba(frame[core.MODEL_FEATURES])[0]
    by_class = dict(zip(model.classes_, probs))
    return {label: float(by_class.get(label, 0.0)) for label in core.LABELS}


# ===================================================================== the window
class FatigueApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Wearable Fatigue Risk Checker")
        self.root.geometry("1180x%d" % min(790, self.root.winfo_screenheight() - 90))
        self.root.minsize(1050, 640)
        self.root.configure(bg=BG)

        self.model, self.model_note = load_model()
        self.vars = {}                 # key -> DoubleVar of its slider
        self.value_labels = {}         # key -> Label showing the number
        self.specs = {}                # key -> (label, lo, hi, step, default, fmt)
        self.update_job = None

        self.setup_style()
        self.build_header()
        self.build_inputs()
        self.build_results()
        self.apply_preset("Rested day worker")

    # ------------------------------------------------------------------ style
    def setup_style(self):
        for name in ("TkDefaultFont", "TkTextFont"):
            tkfont.nametofont(name).configure(size=10)
        self.bold = tkfont.nametofont("TkDefaultFont").copy()
        self.bold.configure(weight="bold")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background=BG)
        style.configure("TLabel", background=BG)
        style.configure("TLabelframe", background=BG, bordercolor="#cbd5e1")
        style.configure("TLabelframe.Label", background=BG, foreground="#0f172a", font=self.bold)
        style.configure("TButton", padding=(10, 6))
        style.configure("Horizontal.TScale", background=BG)
        for label, colour in RISK_COLOURS.items():
            style.configure(label + ".Horizontal.TProgressbar", background=colour,
                            troughcolor="#e2e8f0", bordercolor="#e2e8f0", thickness=14)

    def build_header(self):
        header = tk.Frame(self.root, bg=HEADER_BG)
        header.pack(fill="x")
        tk.Label(header, text="Wearable Fatigue Risk Checker", bg=HEADER_BG, fg="white",
                 font=("TkDefaultFont", 16, "bold")).pack(side="left", padx=16, pady=10)
        tk.Label(header, text="Heart rate + movement + sleep  ->  possible fatigue risk",
                 bg=HEADER_BG, fg="#94a3b8").pack(side="left", pady=10)

    # ----------------------------------------------------------- left: inputs
    def build_inputs(self):
        body = ttk.Frame(self.root, padding=(12, 10, 6, 6))
        body.pack(side="left", fill="both", expand=True)

        bar = ttk.Frame(body)
        bar.pack(fill="x", pady=(0, 6))
        ttk.Label(bar, text="Example person:", font=self.bold).pack(side="left")
        for name in PRESETS:
            ttk.Button(bar, text=name, command=lambda n=name: self.apply_preset(n)).pack(
                side="left", padx=(6, 0))

        grid = ttk.Frame(body)
        grid.pack(fill="both", expand=True)
        grid.columnconfigure((0, 1), weight=1, uniform="cols")
        for index, (title, items) in enumerate(GROUPS):
            box = ttk.LabelFrame(grid, text=title, padding=8)
            box.grid(row=index // 2, column=index % 2, sticky="nsew", padx=4, pady=4)
            box.columnconfigure(0, weight=1)
            for row, (key, label, lo, hi, step, default, fmt) in enumerate(items):
                self.specs[key] = (label, lo, hi, step, default, fmt)
                var = tk.DoubleVar(value=default)
                self.vars[key] = var
                ttk.Label(box, text=label).grid(row=row * 2, column=0, sticky="w")
                value = ttk.Label(box, text="", font=self.bold, foreground="#2563eb")
                value.grid(row=row * 2, column=1, sticky="e")
                self.value_labels[key] = value
                ttk.Scale(box, from_=lo, to=hi, variable=var,
                          command=lambda _v, k=key: self.on_slider(k)).grid(
                    row=row * 2 + 1, column=0, columnspan=2, sticky="ew", pady=(0, 6))
            grid.rowconfigure(index // 2, weight=1)

    # ---------------------------------------------------------- right: results
    def build_results(self):
        side = ttk.Frame(self.root, padding=(6, 10, 12, 6), width=430)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)

        self.badge = tk.Label(side, text="", font=("TkDefaultFont", 20, "bold"), fg="white",
                              pady=14)
        self.badge.pack(fill="x")
        self.badge_sub = tk.Label(side, text="", font=("TkDefaultFont", 10), fg="white")
        self.badge_sub.pack(fill="x")

        probs = ttk.LabelFrame(side, text="Probability of each risk level", padding=8)
        probs.pack(fill="x", pady=(10, 0))
        probs.columnconfigure(1, weight=1)
        self.bars, self.bar_texts = {}, {}
        for row, label in enumerate(core.LABELS):
            ttk.Label(probs, text=label, width=9).grid(row=row, column=0, sticky="w", pady=2)
            bar = ttk.Progressbar(probs, maximum=100, style=label + ".Horizontal.TProgressbar")
            bar.grid(row=row, column=1, sticky="ew", padx=6)
            text = ttk.Label(probs, text="0 %", width=6, anchor="e")
            text.grid(row=row, column=2)
            self.bars[label], self.bar_texts[label] = bar, text

        why = ttk.LabelFrame(side, text="Risk factors found in the data", padding=8)
        why.pack(fill="x", pady=(10, 0))
        self.why_label = tk.Label(why, text="", justify="left", anchor="nw", bg=BG,
                                  wraplength=370)
        self.why_label.pack(fill="x")

        self.rule_label = ttk.Label(side, text="", wraplength=390, foreground="#475569")
        self.rule_label.pack(fill="x", pady=(8, 0))

        advice = ttk.LabelFrame(side, text="Recommended action", padding=8)
        advice.pack(fill="x", pady=(8, 0))
        self.advice_label = tk.Label(advice, text="", justify="left", anchor="nw", wraplength=370,
                                     bg=BG, font=self.bold)
        self.advice_label.pack(fill="x")

        self.warn_label = tk.Label(side, text="", fg="#b45309", bg=BG, wraplength=390,
                                   justify="left")
        self.warn_label.pack(fill="x", pady=(6, 0))

        ttk.Label(side, text="Model: " + self.model_note, wraplength=390,
                  foreground="#64748b").pack(side="bottom", anchor="w")
        ttk.Label(side, text="Research prototype trained on simulated data - not a medical or "
                             "safety device.", wraplength=390, foreground="#64748b").pack(
            side="bottom", anchor="w", pady=(0, 4))

    # ---------------------------------------------------------------- actions
    def read_values(self):
        """Current slider values, snapped to their step and converted to numbers."""
        values = {}
        for key, var in self.vars.items():
            _label, lo, hi, step, _default, _fmt = self.specs[key]
            number = round(var.get() / step) * step
            number = min(max(number, lo), hi)
            values[key] = int(round(number)) if step >= 1 else round(number, 2)
        return values

    def on_slider(self, key):
        """A slider moved: refresh its number and re-predict (after a short pause)."""
        self.refresh_value_label(key)
        if self.update_job:
            self.root.after_cancel(self.update_job)
        self.update_job = self.root.after(40, self.update_result)

    def refresh_value_label(self, key):
        fmt = self.specs[key][5]
        self.value_labels[key].configure(text=fmt % self.read_values()[key])

    def apply_preset(self, name):
        """Fill every slider with an example person."""
        changes = PRESETS[name]
        for key, spec in self.specs.items():
            self.vars[key].set(changes.get(key, spec[4]))
            self.refresh_value_label(key)
        self.update_result()

    def update_result(self):
        """Predict and redraw the result panel."""
        self.update_job = None
        values = self.read_values()
        probs = predict(self.model, values)
        level = max(probs, key=probs.get)

        self.badge.configure(text="FATIGUE RISK: " + level.upper(), bg=RISK_COLOURS[level])
        self.badge_sub.configure(text="%.0f %% confidence" % (probs[level] * 100),
                                 bg=RISK_COLOURS[level])
        for label in core.LABELS:
            self.bars[label]["value"] = probs[label] * 100
            self.bar_texts[label].configure(text="%.0f %%" % (probs[label] * 100))

        points, reasons = core.rule_score(values)
        if reasons:
            self.why_label.configure(text="\n".join("• " + text for _p, text in reasons[:6]))
        else:
            self.why_label.configure(text="• No risk factors found - all signals look normal.")
        self.rule_label.configure(text="Rule-based second opinion: %d points -> %s risk" % (
            points, core.rule_level(points)))
        self.advice_label.configure(text=core.ADVICE[level], bg=RISK_SOFT[level])

        warnings = []
        if values["shift_hours"] > values["hours_awake"]:
            warnings.append("Check: hours on shift cannot be more than hours awake.")
        if values["hrv_rmssd"] > 2.5 * values["hrv_baseline"]:
            warnings.append("Check: HRV is unusually high compared with the baseline.")
        self.warn_label.configure(text="\n".join(warnings))


def main():
    root = tk.Tk()
    FatigueApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

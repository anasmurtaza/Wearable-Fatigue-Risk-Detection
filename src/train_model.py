# -*- coding: utf-8 -*-
"""
train_model.py - train, compare and save the fatigue-risk models
================================================================
Steps
  1. Load the dataset and add the engineered features.
  2. Split BY PERSON (GroupShuffleSplit): nobody appears in both train and test,
     so the test score shows how the model works for NEW people.
  3. Compare an explainable rule-based baseline with three ML models
     (Logistic Regression, Random Forest, Gradient Boosting). The best ML model
     is chosen with 5-fold GroupKFold cross-validation on the TRAINING people only.
  4. Evaluate once on the held-out test people (accuracy, macro-F1, recall of the
     "High" class - missing a really fatigued person is the costly mistake).
  5. Save figures, tables and the final model (re-fitted on all data).

Run:  python train_model.py
"""

import json
import sys
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")                       # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, recall_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import fatigue_core as core

SEED = 42
COLORS = {"Low": "#16a34a", "Moderate": "#f59e0b", "High": "#dc2626"}


def make_models():
    """The three machine-learning candidates."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.08, random_state=SEED),
    }


def load_data():
    """Read the CSV and add the derived features."""
    return core.engineer(pd.read_csv(core.DATA_FILE))


def scores(y_true, y_pred):
    """The metrics we report for every model."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=core.LABELS),
        "high_recall": recall_score(y_true, y_pred, labels=["High"], average=None)[0],
        "high_precision": ((np.array(y_pred) == "High") & (np.array(y_true) == "High")).sum()
                          / max(1, (np.array(y_pred) == "High").sum()),
    }


def fit_final_model(df, name="Random Forest"):
    """Fit the chosen model on ALL data (used for deployment and by the GUI fallback)."""
    model = make_models()[name]
    model.fit(df[core.MODEL_FEATURES], df["fatigue_risk"])
    return model


def train_and_evaluate(verbose=True):
    """Run the whole experiment and return everything the notebook needs."""
    df = load_data()
    X, y, groups = df[core.MODEL_FEATURES], df["fatigue_risk"], df["subject_id"]

    # ---- 2. person-wise split
    split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_idx, test_idx = next(split.split(X, y, groups))
    X_tr, X_te, y_tr, y_te = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    g_tr = groups.iloc[train_idx]

    # ---- 3. rule baseline + cross-validated ML models
    rule_pred = np.array([core.rule_level(p) for p in core.rule_points(df.iloc[test_idx])])
    results = {"Rule-based score": scores(y_te, rule_pred)}
    predictions = {"Rule-based score": rule_pred}
    cv_scores, fitted = {}, {}
    for name, model in make_models().items():
        cv = cross_val_score(model, X_tr, y_tr, groups=g_tr, cv=GroupKFold(n_splits=5),
                             scoring="f1_macro", n_jobs=1)
        cv_scores[name] = cv
        model.fit(X_tr, y_tr)
        fitted[name] = model
        predictions[name] = model.predict(X_te)
        results[name] = scores(y_te, predictions[name])
        if verbose:
            print("%-20s CV macro-F1 %.3f +/- %.3f | test acc %.3f  macro-F1 %.3f  High recall %.3f" % (
                name, cv.mean(), cv.std(), results[name]["accuracy"],
                results[name]["macro_f1"], results[name]["high_recall"]))

    best = max(cv_scores, key=lambda n: cv_scores[n].mean())     # chosen on training people only
    table = pd.DataFrame(results).T.round(3)
    table.insert(0, "cv_macro_f1", [round(cv_scores[n].mean(), 3) if n in cv_scores else np.nan
                                   for n in table.index])
    if verbose:
        print("\nBest model by cross-validation:", best)

    # ---- importance of every signal (permutation, on unseen people)
    imp = permutation_importance(fitted[best], X_te, y_te, scoring="f1_macro",
                                 n_repeats=10, random_state=SEED, n_jobs=1)
    importance = pd.Series(imp.importances_mean, index=core.MODEL_FEATURES).sort_values()

    return {"df": df, "table": table, "best": best, "fitted": fitted, "y_test": y_te,
            "X_train": X_tr, "X_test": X_te, "y_train": y_tr,
            "predictions": predictions, "importance": importance, "cv_scores": cv_scores,
            "n_train": len(train_idx), "n_test": len(test_idx),
            "subjects_train": int(g_tr.nunique()),
            "subjects_test": int(groups.iloc[test_idx].nunique())}


# ---------------------------------------------------------------- signal groups
GROUPS = {
    "Sleep": ["sleep_hours", "sleep_efficiency", "awakenings", "sleep_debt_7d"],
    "Schedule + body clock": ["hours_awake", "shift_hours", "circadian_risk"],
    "Heart rate + HRV": ["hr_mean", "hr_delta", "hrv_ratio"],
    "Movement": ["activity_level", "idle_min_per_hr", "movement_var"],
}


def signal_group_comparison(res):
    """Which kind of signal is useful on its own? Same model, different inputs."""
    experiments = {
        "Heart rate + HRV only": GROUPS["Heart rate + HRV"],
        "Movement only": GROUPS["Movement"],
        "Heart + Movement (body signals)": GROUPS["Heart rate + HRV"] + GROUPS["Movement"],
        "Sleep only": GROUPS["Sleep"],
        "Sleep + Schedule": GROUPS["Sleep"] + GROUPS["Schedule + body clock"],
        "All signals": core.MODEL_FEATURES,
    }
    rows = {}
    for name, cols in experiments.items():
        model = make_models()[res["best"]]
        model.fit(res["X_train"][cols], res["y_train"])
        pred = model.predict(res["X_test"][cols])
        rows[name] = scores(res["y_test"], pred)
    return pd.DataFrame(rows).T.round(3)


# ---------------------------------------------------------------- figures
def plot_confusion(ax, y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=core.LABELS)
    pct = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, "%d\n(%.0f%%)" % (cm[i, j], pct[i, j] * 100), ha="center", va="center",
                    color="white" if pct[i, j] > 0.55 else "#0f172a", fontsize=9)
    ax.set_xticks(range(3), core.LABELS)
    ax.set_yticks(range(3), core.LABELS)
    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("True risk")
    ax.set_title(title, fontsize=10, fontweight="bold")


def save_figures(res):
    """Write the evaluation figures to results/figures."""
    core.FIG_DIR.mkdir(parents=True, exist_ok=True)
    y_te, best = res["y_test"], res["best"]

    # 1. confusion matrices: rule baseline vs best ML model
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    plot_confusion(axes[0], y_te, res["predictions"]["Rule-based score"], "Rule-based score")
    plot_confusion(axes[1], y_te, res["predictions"][best], best)
    fig.suptitle("Confusion matrices on unseen people", fontsize=11)
    fig.tight_layout()
    fig.savefig(core.FIG_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)

    # 2. model comparison
    table = res["table"]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.26
    x = np.arange(len(table))
    for k, (col, label, colour) in enumerate((("accuracy", "Accuracy", "#2563eb"),
                                              ("macro_f1", "Macro-F1", "#16a34a"),
                                              ("high_recall", "Recall of 'High'", "#dc2626"))):
        bars = ax.bar(x + (k - 1) * width, table[col], width, label=label, color=colour)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)
    ax.set_xticks(x, table.index, fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_title("Model comparison on unseen people")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(core.FIG_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)

    # 3. which signals matter
    imp = res["importance"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh([core.PRETTY[n] for n in imp.index], imp.values, color="#2563eb")
    ax.set_xlabel("Drop in macro-F1 when the signal is shuffled")
    ax.set_title("Which signals drive the prediction (%s)" % res["best"])
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(core.FIG_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)


def plot_signal_groups(groups):
    """Bar chart of macro-F1 for each input set."""
    fig, ax = plt.subplots(figsize=(8, 4))
    colours = ["#f59e0b", "#f59e0b", "#d97706", "#2563eb", "#1d4ed8", "#16a34a"]
    bars = ax.barh(groups.index[::-1], groups["macro_f1"][::-1], color=colours[::-1])
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro-F1 on unseen people (same model, different inputs)")
    ax.set_title("What each kind of signal can do on its own")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def main():
    res = train_and_evaluate()
    core.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    core.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_figures(res)
    groups = signal_group_comparison(res)
    groups.to_csv(core.RESULTS_DIR / "signal_group_comparison.csv")
    fig = plot_signal_groups(groups)
    fig.savefig(core.FIG_DIR / "signal_groups.png", dpi=150)
    plt.close(fig)
    print("\n" + groups.to_string())
    res["table"].to_csv(core.RESULTS_DIR / "model_comparison.csv")
    report = classification_report(res["y_test"], res["predictions"][res["best"]],
                                   labels=core.LABELS, digits=3)
    (core.RESULTS_DIR / "classification_report.txt").write_text(
        "Best model: %s (test people: %d, windows: %d)\n\n%s" % (
            res["best"], res["subjects_test"], res["n_test"], report), encoding="utf-8")
    print("\n" + report)

    final = fit_final_model(res["df"], res["best"])      # deployment model: all data
    joblib.dump({"model": final, "features": core.MODEL_FEATURES, "labels": core.LABELS,
                 "name": res["best"], "sklearn": sklearn.__version__}, core.MODEL_FILE)
    info = {"model": res["best"], "trained": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python": sys.version.split()[0], "sklearn": sklearn.__version__,
            "train_windows": res["n_train"], "test_windows": res["n_test"],
            "train_people": res["subjects_train"], "test_people": res["subjects_test"],
            "test_scores": res["table"].loc[res["best"]].to_dict()}
    (core.MODEL_DIR / "model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("Saved model to", core.MODEL_FILE)


if __name__ == "__main__":
    main()

"""Draw the two overview charts used in the root README, from the saved result files.

Run from the project folder (needs pandas and matplotlib; the project .venv has both):
    python docs/make_overview_figures.py

Reads (nothing is re-computed from images, and no model is run):
    approaches/3_deep_learning/results/test_metrics.csv          full test set, three approaches
    approaches/3_deep_learning/results/human_subset_metrics.csv  the 40-image subset, three approaches
    eval/ground_truth.csv and eval/predictions/*.csv             confusion counts for the error chart
Writes:
    docs/figures/approach_comparison.png
    docs/figures/error_breakdown.png

Colours: the deep learning red and symbolic purple match the notebook figures; the human baseline is teal here
(the notebook uses grey, which is too close to the purple for people with colour-vision differences).
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "approaches" / "3_deep_learning" / "results"
OUT = Path(__file__).resolve().parent / "figures"

COLOR = {"Human": "#2A9D8F", "Symbolic rule": "#7F6BB5", "Deep learning": "#C44E52", "Always positive": "#D9D9D9"}
INK, MUTED = "#222222", "#666666"
METRICS = [("F1", "F1"), ("Recall", "Recall"), ("Specificity", "Specificity"), ("BalAcc", "Balanced\naccuracy"), ("MCC", "MCC")]

plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.axisbelow": True, "grid.alpha": 0.25, "font.size": 10,
                     "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK})


def grouped_bars(ax, table, approaches, title):
    """One group per metric, one bar per approach, value printed above each bar."""
    width = 0.8 / len(approaches)
    for k, name in enumerate(approaches):
        xs = [i + (k - (len(approaches) - 1) / 2) * width for i in range(len(METRICS))]
        vals = [float(table.loc[name, col]) for col, _ in METRICS]
        ax.bar(xs, vals, width * 0.92, color=COLOR[name], label=name,
               hatch="///" if name == "Always positive" else None,
               edgecolor="#9A9A9A" if name == "Always positive" else "none", linewidth=0)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(METRICS)), [label for _, label in METRICS])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score (higher is better)")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="x", visible=False)


def comparison_figure():
    full = pd.read_csv(RESULTS / "test_metrics.csv", index_col=0)
    subset = pd.read_csv(RESULTS / "human_subset_metrics.csv", index_col=0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    grouped_bars(a, full, ["Always positive", "Symbolic rule", "Deep learning"], "Full test set (624 images)")
    grouped_bars(b, subset, ["Human", "Symbolic rule", "Deep learning"], "40-image subset a person labelled")
    for ax in (a, b):
        ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    fig.suptitle("Same task, same images, three ways to answer: PNEUMONIA or NORMAL?", x=0.01, ha="left",
                 fontsize=12.5, fontweight="bold")
    fig.text(0.01, 0.005, "Positive class = PNEUMONIA. Always positive = answers PNEUMONIA for every image (the trivial baseline).",
             fontsize=8, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(OUT / "approach_comparison.png")
    plt.close(fig)


def read_labels(path, label_col):
    with open(path, newline="") as f:
        return {r["Example ID"]: r[label_col] for r in csv.DictReader(f)}


def error_figure():
    truth = read_labels(PROJECT / "eval" / "ground_truth.csv", "Ground Truth Label")
    preds = {
        "Always positive": {i: "Positive" for i in truth},
        "Symbolic rule": read_labels(PROJECT / "eval" / "predictions" / "symbolic_eval.csv", "Algorithmic Prediction"),
        "Deep learning": read_labels(PROJECT / "eval" / "predictions" / "deep_learning_eval.csv", "Deep Learning Prediction"),
    }
    n_pos = sum(v == "Positive" for v in truth.values())
    n_neg = len(truth) - n_pos
    counts = {}
    for name, p in preds.items():
        fn = sum(truth[i] == "Positive" and p[i] == "Negative" for i in truth)
        fp = sum(truth[i] == "Negative" and p[i] == "Positive" for i in truth)
        counts[name] = (fn, fp)

    names = list(preds)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.4), sharey=True)
    for ax, idx, total, title in ((a, 0, n_pos, f"Missed pneumonia (out of {n_pos} sick children)"),
                                  (b, 1, n_neg, f"False alarms (out of {n_neg} healthy children)")):
        ys = range(len(names))[::-1]
        for y, name in zip(ys, names):
            c = counts[name][idx]
            ax.barh(y, c / total * 100, 0.6, color=COLOR[name], hatch="///" if name == "Always positive" else None,
                    edgecolor="#9A9A9A" if name == "Always positive" else "none", linewidth=0)
            ax.text(c / total * 100 + 1.5, y, f"{c} ({c / total:.0%})", va="center", fontsize=9)
        ax.set_yticks(list(ys), names)
        ax.set_xlim(0, 118)
        ax.set_xlabel("% of that group answered wrongly (lower is better)")
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Two different kinds of mistake, on the full test set", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "error_breakdown.png")
    plt.close(fig)
    return counts


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    comparison_figure()
    print("error counts (missed pneumonia, false alarms):", error_figure())
    print(f"wrote {OUT / 'approach_comparison.png'} and {OUT / 'error_breakdown.png'}")

"""Draw how the three symbolic-rule features separate NORMAL from PNEUMONIA, on the TRAINING split only.

Run from the project folder (needs pandas and matplotlib):
    python approaches/2_symbolic/plot_feature_distributions.py

Reads  approaches/2_symbolic/train_features.csv  (cache built by explore_features.py; rows with part == "train").
Writes approaches/2_symbolic/figures/feature_distributions.png
The test set is never read. Thresholds and the vote rule are imported from symbolic.py, so the picture always shows
the rule that is actually used.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import symbolic  # noqa: E402

NORMAL, PNEU, INK, MUTED = "#4C72B0", "#DD8452", "#222222", "#666666"
TITLES = {
    "lung_edges": ("Edge density in the lungs", "pneumonia side: LOWER (hazier, fewer sharp edges)"),
    "lung_detail": ("Fine detail in the lungs", "pneumonia side: LOWER (less rib/vessel contrast)"),
    "lung_mean": ("Average lung brightness", "pneumonia side: HIGHER (whiter lungs)"),
}

plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "axes.axisbelow": True, "grid.alpha": 0.25, "font.size": 10, "text.color": INK,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})

if __name__ == "__main__":
    df = pd.read_csv(HERE / "train_features.csv")
    df = df[df.part == "train"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, (feat, (title, note)) in zip(axes, TITLES.items()):
        lo, hi = df[feat].quantile([0.005, 0.995])
        bins = np.linspace(lo, hi, 40)
        for label, color, name in ((0, NORMAL, "NORMAL"), (1, PNEU, "PNEUMONIA")):
            ax.hist(df.loc[df.label == label, feat], bins=bins, density=True, alpha=0.65, color=color, label=name, linewidth=0)
        threshold, _ = symbolic.THRESHOLDS[feat]
        ax.axvline(threshold, color=INK, ls="--", lw=1.4)
        ax.text(threshold, ax.get_ylim()[1] * 0.97, f" threshold {threshold:.2f}", fontsize=8.5, va="top", ha="left")
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
        ax.set_xlabel(note, fontsize=8, color=MUTED)
        ax.set_yticks([])
        ax.set_ylabel("share of images" if feat == "lung_edges" else "")
        if feat == "lung_edges":
            ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.0, 0.8))

    # fourth panel: how many features vote "pneumonia" vs how many of those images really have pneumonia
    votes = np.zeros(len(df), dtype=int)
    for feat, (threshold, sign) in symbolic.THRESHOLDS.items():
        votes += ((df[feat].to_numpy() - threshold) * sign > 0).astype(int)
    ax = axes[3]
    shares = [df.label[votes == k].mean() * 100 for k in range(4)]
    counts = [(votes == k).sum() for k in range(4)]
    bars = ax.bar(range(4), shares, 0.62, color=PNEU, linewidth=0)
    for x, s, n in zip(range(4), shares, counts):
        ax.text(x, s + 2, f"{s:.0f}%\n(n={n})", ha="center", va="bottom", fontsize=8.5)
    ax.axhline(df.label.mean() * 100, color=MUTED, ls=":", lw=1)
    ax.text(-0.45, df.label.mean() * 100 + 1.5, f"all training images: {df.label.mean():.0%} pneumonia", fontsize=8,
            color=MUTED, ha="left")
    ax.axvline(symbolic.VOTES_NEEDED - 0.5, color=INK, ls="--", lw=1.4)
    ax.text(symbolic.VOTES_NEEDED - 0.45, 6, "answer PNEUMONIA\nfrom here on", fontsize=8.5, va="bottom")
    ax.set_xticks(range(4), ["0", "1", "2", "3"])
    ax.set_xlabel("number of features voting 'pneumonia'", fontsize=9)
    ax.set_ylim(0, 118)
    ax.set_ylabel("% truly pneumonia")
    ax.set_title("The vote", loc="left", fontsize=10.5, fontweight="bold")
    ax.grid(axis="x", visible=False)

    fig.suptitle(f"Symbolic rule on the {len(df):,} training images: three measurements vote, 2 of 3 decides",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "feature_distributions.png")
    print("vote buckets (votes -> n, % pneumonia):", {k: (int(c), round(s, 1)) for k, (c, s) in enumerate(zip(counts, shares))})
    print(f"wrote {out / 'feature_distributions.png'}")

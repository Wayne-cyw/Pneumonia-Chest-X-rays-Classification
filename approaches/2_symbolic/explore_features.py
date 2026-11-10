"""Measure the candidate features of symbolic.py on the TRAINING data only.

Uses the project split in data/image_index.csv (column `split`): the `train` part is for rule design and
the `val` part is a check on images the rule was not tuned on. The test set is never read.

Run from the project folder:  python approaches/2_symbolic/explore_features.py
Writes approaches/2_symbolic/train_features.csv (cache) and prints how well each feature separates the classes.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score  # used only to measure a feature, not to train a model

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "approaches" / "2_symbolic"))
import symbolic  # noqa: E402

CACHE = PROJECT / "approaches" / "2_symbolic" / "train_features.csv"


def train_rows():
    rows = [r for r in csv.DictReader(open(PROJECT / "data" / "image_index.csv")) if r["split"] in ("train", "val")]
    assert rows and all(r["split"] in ("train", "val") for r in rows)  # guard: never touch the test set
    return rows


def build_cache():
    rows = train_rows()
    out = []
    for r in rows:
        name = Path(r["path"]).name
        feats = symbolic.extract_features(PROJECT / r["path"])
        out.append({
            "Example ID": r["Example ID"],
            "part": r["split"],
            "label": 1 if "/PNEUMONIA/" in r["path"] else 0,
            "subtype": "bacteria" if "_bacteria_" in name else ("virus" if "_virus_" in name else "normal"),
            **feats,
        })
    CACHE.parent.mkdir(exist_ok=True)
    with open(CACHE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    return out


def load_cache():
    with open(CACHE) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"] = int(r["label"])
        for k in r:
            if k.startswith("lung"):
                r[k] = float(r[k])
    return rows


if __name__ == "__main__":
    rows = load_cache() if CACHE.exists() and "--rebuild" not in sys.argv else build_cache()
    features = [k for k in rows[0] if k.startswith("lung")]
    for part in ("train", "val"):
        sub = [r for r in rows if r["part"] == part]
        pos = sum(r["label"] for r in sub)
        print(f"\n{part}: {len(sub)} images, {pos} pneumonia ({100 * pos / len(sub):.1f}%)")
        y = np.array([r["label"] for r in sub])
        print(f"  {'feature':16s} {'AUC':>6s}  {'median NORMAL':>14s} {'median bacteria':>16s} {'median virus':>13s}")
        for feat in features:
            x = np.array([r[feat] for r in sub])
            auc = roc_auc_score(y, x)
            med = {s: np.median([r[feat] for r in sub if r["subtype"] == s]) for s in ("normal", "bacteria", "virus")}
            print(f"  {feat:16s} {auc:6.3f}  {med['normal']:14.3f} {med['bacteria']:16.3f} {med['virus']:13.3f}")

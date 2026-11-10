"""Choose the thresholds and the voting rule for symbolic.py using TRAINING data only.

Uses approaches/2_symbolic/train_features.csv from explore_features.py: the "dev" part sets the thresholds,
the "val" part (different patients, also from the training folder) checks them.
The test set is not read.

Three features vote; each votes "Positive" (pneumonia) when it is on the pneumonia side of its threshold:
    lung_edges  < threshold  (hazy lungs have fewer sharp edges)
    lung_detail < threshold  (less fine rib/vessel contrast)
    lung_mean   > threshold  (brighter lungs)
Two ways of setting each threshold are compared:
    midpoint : halfway between the NORMAL median and the PNEUMONIA median on dev (easy to state by hand)
    balanced : the cut that maximises balanced accuracy on dev (ignores the class imbalance)
"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_features import load_cache  # noqa: E402

# feature -> +1 if larger values mean pneumonia, -1 if smaller values mean pneumonia
FEATURES = {"lung_edges": -1, "lung_detail": -1, "lung_mean": +1}


def metrics(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    recall = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1); prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * recall / max(prec + recall, 1e-9)
    return {"F1": f1, "recall": recall, "specificity": spec, "bal_acc": (recall + spec) / 2,
            "acc": (tp + tn) / len(y), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def votes(rows, thresholds):
    out = np.zeros(len(rows), dtype=int)
    for feat, sign in FEATURES.items():
        x = np.array([r[feat] for r in rows])
        out += ((x - thresholds[feat]) * sign > 0).astype(int)
    return out


def midpoint_threshold(x, y):
    return (np.median(x[y == 0]) + np.median(x[y == 1])) / 2


def balanced_threshold(x, y, sign):
    best, best_t = -1, None
    for t in np.unique(np.quantile(x, np.linspace(0.01, 0.99, 197))):
        pred = ((x - t) * sign > 0).astype(int)
        m = metrics(y, pred)["bal_acc"]
        if m > best:
            best, best_t = m, t
    return best_t


if __name__ == "__main__":
    rows = load_cache()
    dev = [r for r in rows if r["part"] == "dev"]; val = [r for r in rows if r["part"] == "val"]
    y_dev = np.array([r["label"] for r in dev]); y_val = np.array([r["label"] for r in val])
    for method in ("midpoint", "balanced"):
        thr = {}
        for feat, sign in FEATURES.items():
            x = np.array([r[feat] for r in dev])
            cut = midpoint_threshold(x, y_dev) if method == "midpoint" else balanced_threshold(x, y_dev, sign)
            thr[feat] = round(float(cut), 2)  # cutoffs are kept to 2 decimal places, as in symbolic.py
        print(f"\n=== thresholds: {method} (rounded to 2 decimals) ===")
        print(thr)
        for k in (1, 2, 3):
            for name, part, y in (("dev", dev, y_dev), ("val", val, y_val)):
                m = metrics(y, (votes(part, thr) >= k).astype(int))
                print(f"  >= {k} of 3 votes  {name}: F1 {m['F1']:.3f}  recall {m['recall']:.3f}  "
                      f"spec {m['specificity']:.3f}  bal_acc {m['bal_acc']:.3f}  acc {m['acc']:.3f}  "
                      f"(tp {m['tp']} fp {m['fp']} fn {m['fn']} tn {m['tn']})")

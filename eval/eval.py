import csv
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
EVAL = PROJECT / "eval"
LABELS = ("Positive", "Negative")


# Read one CSV into {example id: label}, exiting with a clear error if it is not usable.
def read_column(path, id_col, label_col):
    if not path.exists():
        sys.exit(f"ERROR: {path} not found.")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for col in (id_col, label_col):
            if col not in (reader.fieldnames or []):
                sys.exit(f"ERROR: {path.name} has no column '{col}' (found {reader.fieldnames}).")
        values = {}
        for row in reader:
            example_id, label = row[id_col].strip(), row[label_col].strip()
            if example_id in values:
                sys.exit(f"ERROR: {path.name} lists {example_id} more than once.")
            if label not in LABELS:
                sys.exit(f"ERROR: {path.name}: {example_id} has label {label!r}; expected one of {LABELS}.")
            values[example_id] = label
    return values


# Count TP, FP, FN and TN over the given IDs, with Positive as the positive class.
def confusion(truth, pred, ids):
    counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for i in ids:
        actual_pos, pred_pos = truth[i] == "Positive", pred[i] == "Positive"
        counts[("T" if actual_pos == pred_pos else "F") + ("P" if pred_pos else "N")] += 1
    return counts


# Return num / den, or None when the denominator is 0.
def ratio(num, den):
    return num / den if den else None


# Compute F1, recall, specificity, precision, accuracy and balanced accuracy from confusion counts.
def metrics(c):
    precision = ratio(c["TP"], c["TP"] + c["FP"])
    recall = ratio(c["TP"], c["TP"] + c["FN"])
    specificity = ratio(c["TN"], c["TN"] + c["FP"])
    f1 = ratio(2 * c["TP"], 2 * c["TP"] + c["FP"] + c["FN"])
    balanced = None if recall is None or specificity is None else (recall + specificity) / 2
    return {
        "F1": f1,
        "Recall": recall,
        "Specificity": specificity,
        "Precision": precision,
        "Accuracy": ratio(c["TP"] + c["TN"], sum(c.values())),
        "Balanced accuracy": balanced,
    }


# Format a metric to 3 decimals, or "undefined" when it could not be computed.
def fmt(value):
    return "undefined" if value is None else f"{value:.3f}"


# Print n, the confusion counts and all metrics for one approach on one set of IDs.
def report(name, truth, pred, ids):
    c = confusion(truth, pred, ids)
    print(f"{name}  (n = {len(ids)})")
    print(f"  confusion: TP={c['TP']}  FP={c['FP']}  FN={c['FN']}  TN={c['TN']}")
    for metric, value in metrics(c).items():
        print(f"  {metric:<18}{fmt(value)}")
    print()


# Describe how many of the given IDs are Positive in the ground truth.
def pneumonia_share(truth, ids):
    positives = sum(truth[i] == "Positive" for i in ids)
    return f"{positives}/{len(ids)} Positive ({positives / len(ids):.1%})"


if __name__ == "__main__":
    human_path = Path(sys.argv[1]) if len(sys.argv) > 1 else EVAL / "predictions" / "human_eval.csv"

    truth = read_column(EVAL / "ground_truth.csv", "Example ID", "Ground Truth Label")
    algo = read_column(EVAL / "predictions" / "symbolic_eval.csv", "Example ID", "Algorithmic Prediction")
    human = read_column(human_path, "Example ID", "Human Prediction")

    if set(algo) != set(truth):
        sys.exit(f"ERROR: symbolic_eval.csv and ground_truth.csv differ: "
                 f"{len(set(truth) - set(algo))} test IDs have no prediction, "
                 f"{len(set(algo) - set(truth))} predicted IDs are not in the ground truth.")
    if not set(human) <= set(truth):
        sys.exit(f"ERROR: {human_path.name} has IDs not in ground_truth.csv: {sorted(set(human) - set(truth))}")

    test_ids = sorted(truth)
    subset_ids = sorted(human)

    print("Positive = PNEUMONIA, Negative = NORMAL; F1/recall/precision are for Positive.\n")
    print("Class balance")
    print(f"  full test set : {pneumonia_share(truth, test_ids)}")
    print(f"  human subset  : {pneumonia_share(truth, subset_ids)}\n")

    report("Human baseline, human subset", truth, human, subset_ids)
    report("Algorithm, human subset", truth, algo, subset_ids)
    report("Algorithm, full test set", truth, algo, test_ids)

    disagree = [i for i in subset_ids if human[i] != algo[i]]
    both_right = sum(human[i] == algo[i] == truth[i] for i in subset_ids)
    only_human = sum(human[i] == truth[i] != algo[i] for i in subset_ids)
    only_algo = sum(algo[i] == truth[i] != human[i] for i in subset_ids)
    print(f"Human vs. algorithm on the human subset (n = {len(subset_ids)})")
    print(f"  disagree on {len(disagree)} images; both correct on {both_right}; "
          f"only human correct on {only_human}; only algorithm correct on {only_algo}")
    print(f"  disagreeing IDs: {', '.join(disagree) if disagree else 'none'}")

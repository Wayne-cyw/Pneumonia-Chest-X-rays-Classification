"""Score eval/human_baseline.csv against eval/ground_truth.csv.

Run this only AFTER you have finished labelling and exported the CSV
(Lab 2: don't compare against ground truth before the baseline is final).
"""
import csv
import sys
from pathlib import Path

EVAL = Path(__file__).resolve().parent.parent / "eval"
human_path = Path(sys.argv[1]) if len(sys.argv) > 1 else EVAL / "predictions" / "human_eval.csv"

truth = {r["Example ID"]: r["Ground Truth Label"] for r in csv.DictReader(open(EVAL / "ground_truth.csv"))}
human = {r["Example ID"]: r["Human Prediction"] for r in csv.DictReader(open(human_path))}

missing = [k for k, v in human.items() if v not in ("Positive", "Negative")]
if missing:
    sys.exit(f"Unlabelled or invalid rows: {missing}")

correct = sum(human[k] == truth[k] for k in human)
print(f"Human accuracy: {correct}/{len(human)} = {correct / len(human):.1%}")

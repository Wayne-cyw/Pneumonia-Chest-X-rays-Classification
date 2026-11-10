"""Run symbolic.py ONCE on the full test set and write eval/predictions/symbolic_eval.csv.

Do not change the rule after seeing test results. Requires --confirm so it is never run by accident.

    python approaches/2_symbolic/make_symbolic_eval.py --confirm
"""
import csv
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "approaches" / "2_symbolic"))
import symbolic  # noqa: E402

if __name__ == "__main__":
    if "--confirm" not in sys.argv:
        sys.exit("Refusing to run on the test set without --confirm.")
    rows = [r for r in csv.DictReader(open(PROJECT / "data" / "image_index.csv")) if r["split"] == "test"]
    out = PROJECT / "eval" / "predictions" / "symbolic_eval.csv"
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Example ID", "Algorithmic Prediction"])
        for r in sorted(rows, key=lambda r: r["Example ID"]):
            writer.writerow([r["Example ID"], symbolic.predict(PROJECT / r["path"])])
    print(f"wrote {len(rows)} predictions to {out}")

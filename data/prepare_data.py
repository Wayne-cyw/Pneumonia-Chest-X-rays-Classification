"""Set up the project data from the raw Kaggle chest X-ray download.

Run once from the project folder:  python data/prepare_data.py     (needs scikit-learn)

Reads:
  data/raw/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/

Writes:
  data/image_index.csv   one row per image: Example ID, path, label, patient_id, kaggle_split, split
  eval/ground_truth.csv  Example ID, Ground Truth Label   (test images only; Positive = PNEUMONIA)
  data/data_checks.txt   split sizes, class balance, duplicate and patient-overlap checks

The one project-wide split lives in the `split` column of image_index.csv:
  test   = the dataset's own test folder, untouched
  train / val = an 80/20 split of the dataset's train + val folders, grouped by patient and stratified by class
Every approach reads this column; nothing else splits the data.

Patients: pneumonia file names carry a patient number (person<N>_...), so those images are grouped by it.
Normal file names carry no usable patient ID (the IM-xxxx number is a coincidence), so each normal image
is its own group; the split cannot stop one patient's normal X-rays from landing on both sides.

Example IDs are assigned after a seeded shuffle so the ID order does not reveal the class. The shuffle
order (train, then val, then test, one Random(SEED)) must not change: every ID in eval/ depends on it.
"""
import csv
import hashlib
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

PROJECT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT / "data" / "raw" / "chest_xray"  # Kaggle layout: {train,val,test}/{NORMAL,PNEUMONIA}
SEED = 211        # Example-ID shuffle
SPLIT_SEED = 211  # train/val split (separate from the ID shuffle so the IDs never move)
N_SPLITS = 5      # fold 0 of 5 puts about 20% of the patient groups into val
POSITIVE, NEGATIVE = "Positive", "Negative"  # Positive = PNEUMONIA, Negative = NORMAL
LABEL_OF_FOLDER = {"PNEUMONIA": POSITIVE, "NORMAL": NEGATIVE}
PERSON = re.compile(r"person(\d+)_")  # person123_bacteria_456.jpeg


def list_images(split):
    """Return (path, folder_name) for every image in a Kaggle split folder, in a stable order."""
    items = []
    for folder in sorted(LABEL_OF_FOLDER):
        for path in sorted((DATA_ROOT / split / folder).glob("*.jp*g")):
            items.append((path, folder))
    return items


def patient_id(filename, label, example_id):
    """Pneumonia: P<N> from person<N>_... (bacterial and viral files with the same N are one patient).
    Normal, or an unparsable pneumonia name: a group of one image."""
    if label == POSITIVE:
        match = PERSON.match(filename)
        return f"P{match.group(1)}" if match else f"P_{example_id}"
    return f"N_{example_id}"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    if not DATA_ROOT.exists():
        raise SystemExit(f"Dataset not found at {DATA_ROOT}")
    rng = random.Random(SEED)
    kaggle = {name: list_images(name) for name in ("train", "val", "test")}

    # Assign Example IDs after shuffling, so ID order carries no class information.
    rows = []
    for kaggle_split, prefix in (("train", "train"), ("val", "val"), ("test", "ex")):
        items = kaggle[kaggle_split][:]
        rng.shuffle(items)
        width = max(len(str(len(items))), 3)
        for i, (path, folder) in enumerate(items, start=1):
            example_id = f"{prefix}_{i:0{width}d}"
            label = LABEL_OF_FOLDER[folder]
            rows.append({
                "Example ID": example_id,
                "path": path.relative_to(PROJECT).as_posix(),
                "label": 1 if label == POSITIVE else 0,
                "patient_id": patient_id(path.name, label, example_id),
                "kaggle_split": kaggle_split,
                "split": "test" if kaggle_split == "test" else None,
                "file": path,
            })

    # The project split: Kaggle train + val pooled, 80/20 by patient group, stratified by class.
    pool = [r for r in rows if r["kaggle_split"] != "test"]
    folds = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SPLIT_SEED)
    train_idx, val_idx = next(iter(folds.split(
        np.zeros(len(pool)), [r["label"] for r in pool], [r["patient_id"] for r in pool])))
    for i in train_idx:
        pool[i]["split"] = "train"
    for i in val_idx:
        pool[i]["split"] = "val"
    train_patients = {pool[i]["patient_id"] for i in train_idx}
    val_patients = {pool[i]["patient_id"] for i in val_idx}
    assert not train_patients & val_patients, "patient leakage between train and val"

    write_csv(PROJECT / "data" / "image_index.csv",
              ["Example ID", "path", "label", "patient_id", "kaggle_split", "split"],
              [[r["Example ID"], r["path"], r["label"], r["patient_id"], r["kaggle_split"], r["split"]] for r in rows])

    test_rows = sorted((r for r in rows if r["split"] == "test"), key=lambda r: r["Example ID"])
    write_csv(PROJECT / "eval" / "ground_truth.csv", ["Example ID", "Ground Truth Label"],
              [[r["Example ID"], POSITIVE if r["label"] else NEGATIVE] for r in test_rows])

    # Data checks for the evaluation-setup documentation.
    lines = [f"seed = {SEED} (Example IDs), split seed = {SPLIT_SEED} (train/val, fold 0 of {N_SPLITS})", ""]
    lines.append("Dataset folders as downloaded")
    for name in ("train", "val", "test"):
        counts = Counter(folder for _, folder in kaggle[name])
        total = sum(counts.values())
        lines.append(f"  {name}: {total} images, NORMAL {counts['NORMAL']}, PNEUMONIA {counts['PNEUMONIA']} "
                     f"({100 * counts['PNEUMONIA'] / total:.1f}% pneumonia)")
    test_pneu = Counter("bacteria" if "_bacteria_" in r["file"].name else "virus" for r in test_rows if r["label"])
    lines.append(f"  test PNEUMONIA subtypes: {dict(test_pneu)}")

    lines.append("")
    lines.append("Project split (image_index.csv, column `split`)")
    for name in ("train", "val", "test"):
        part = [r for r in rows if r["split"] == name]
        pos = sum(r["label"] for r in part)
        patients = len({r["patient_id"] for r in part if r["label"]})
        lines.append(f"  {name}: {len(part)} images, NORMAL {len(part) - pos}, PNEUMONIA {pos} "
                     f"({100 * pos / len(part):.1f}% pneumonia), {patients} pneumonia patients")

    hashes = {s: {sha256(p) for p, _ in kaggle[s]} for s in kaggle}
    lines.append("")
    lines.append("Exact duplicate images between dataset folders")
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        lines.append(f"  {a} & {b}: {len(hashes[a] & hashes[b])}")

    lines.append("")
    lines.append("Pneumonia patient IDs shared between splits (normal files have no patient ID, so they cannot be checked)")
    lines.append(f"  project train & val: {len(train_patients & val_patients)} (asserted above)")
    kaggle_patients = {s: {r["patient_id"] for r in rows if r["kaggle_split"] == s and r["label"]} for s in kaggle}
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        lines.append(f"  dataset {a} folder & {b} folder: {len(kaggle_patients[a] & kaggle_patients[b])}")
    lines.append("  (a shared ID may be the same child or two children with the same number; file names cannot tell)")

    report = "\n".join(lines) + "\n"
    (PROJECT / "data" / "data_checks.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()

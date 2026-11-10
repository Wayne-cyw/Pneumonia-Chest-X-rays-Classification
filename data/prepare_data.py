"""Prepare the project data layout (Lab 2 / Lab 3 formats) from the Kaggle chest X-ray dataset.

Run once from the project folder:  python data/prepare_data.py

Produces:
  data/image_index.csv          Example ID, split, path   (path is relative to the project folder)
  eval/ground_truth.csv         Example ID, Ground Truth Label   (test set only; Positive = PNEUMONIA)
  eval/predictions/human_eval_template.csv  Example ID, Human Prediction, Notes   (blank, human subset only)
  approaches/1_human_baseline/images/  the human-subset images, renamed to their Example ID (no label or file name)
  data/sample/                  one test image for the progress report submission
  data/data_checks.txt          split sizes, class balance, duplicate and patient-overlap checks

Example IDs are assigned after a seeded shuffle so the ID order does not reveal the class.
"""
import csv
import hashlib
import random
import re
import shutil
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT / "data" / "raw" / "chest_xray"  # Kaggle layout: {train,val,test}/{NORMAL,PNEUMONIA}
SEED = 211
HUMAN_SUBSET_SIZE = 40
POSITIVE, NEGATIVE = "Positive", "Negative"  # Positive = PNEUMONIA, Negative = NORMAL
LABEL_OF_FOLDER = {"PNEUMONIA": POSITIVE, "NORMAL": NEGATIVE}


def list_images(split):
    """Return (path, folder_name) for every image in a split, in a stable order."""
    items = []
    for folder in sorted(LABEL_OF_FOLDER):
        for path in sorted((DATA_ROOT / split / folder).glob("*.jp*g")):
            items.append((path, folder))
    return items


def patient_key(filename):
    """Best-effort patient ID from a Kaggle file name (used only for the leakage check).

    Bacterial and viral cases are numbered separately (person1_bacteria_1 vs person1_virus_6), and
    NORMAL2-IM-xxxx is a different image series from IM-xxxx, so both parts are kept in the key.
    """
    match = re.match(r"(person\d+)_(bacteria|virus)_", filename)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    match = re.match(r"((?:NORMAL2-)?IM-\d{4})", filename)
    if match:
        return match.group(1)
    return filename


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    rng = random.Random(SEED)
    splits = {name: list_images(name) for name in ("train", "val", "test")}

    # Assign Example IDs after shuffling, so ID order carries no class information.
    meta_rows, records = [], {}
    for split, prefix in (("train", "train"), ("val", "val"), ("test", "ex")):
        items = splits[split][:]
        rng.shuffle(items)
        width = len(str(len(items)))
        width = max(width, 3)
        for i, (path, folder) in enumerate(items, start=1):
            example_id = f"{prefix}_{i:0{width}d}"
            rel = path.relative_to(PROJECT).as_posix()
            meta_rows.append([example_id, split, rel])
            records[example_id] = (split, path, folder)
    write_csv(PROJECT / "data" / "image_index.csv", ["Example ID", "split", "path"], meta_rows)

    # Ground truth for the full test set.
    test_ids = sorted(i for i, (s, _, _) in records.items() if s == "test")
    gt_rows = [[i, LABEL_OF_FOLDER[records[i][2]]] for i in test_ids]
    write_csv(PROJECT / "eval" / "ground_truth.csv", ["Example ID", "Ground Truth Label"], gt_rows)

    # Human evaluation subset: simple random sample of the test set, fixed seed.
    human_ids = sorted(rng.sample(test_ids, HUMAN_SUBSET_SIZE))
    write_csv(
        PROJECT / "eval" / "predictions" / "human_eval_template.csv",
        ["Example ID", "Human Prediction", "Notes"],
        [[i, "", ""] for i in human_ids],
    )
    image_dir = PROJECT / "approaches" / "1_human_baseline" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for i in human_ids:
        src = records[i][1]
        shutil.copyfile(src, image_dir / f"{i}{src.suffix.lower()}")

    # One sample test image for the submission (not from the human subset).
    sample_id = rng.choice([i for i in test_ids if i not in set(human_ids)])
    sample_src = records[sample_id][1]
    (PROJECT / "data" / "sample").mkdir(exist_ok=True)
    shutil.copyfile(sample_src, PROJECT / "data" / "sample" / f"{sample_id}{sample_src.suffix.lower()}")

    # Data checks for the evaluation-setup section.
    lines = [f"seed = {SEED}", f"human subset size = {HUMAN_SUBSET_SIZE}", ""]
    for split in ("train", "val", "test"):
        counts = Counter(folder for _, folder in splits[split])
        total = sum(counts.values())
        lines.append(
            f"{split}: {total} images, NORMAL {counts['NORMAL']}, PNEUMONIA {counts['PNEUMONIA']} "
            f"({100 * counts['PNEUMONIA'] / total:.1f}% pneumonia)"
        )
    test_pneu = Counter(
        "bacteria" if "_bacteria_" in p.name else "virus"
        for p, f in splits["test"] if f == "PNEUMONIA"
    )
    lines.append(f"test PNEUMONIA subtypes: {dict(test_pneu)}")

    human_labels = Counter(LABEL_OF_FOLDER[records[i][2]] for i in human_ids)
    human_sub = Counter(
        "bacteria" if "_bacteria_" in records[i][1].name else "virus"
        for i in human_ids if records[i][2] == "PNEUMONIA"
    )
    lines.append("")
    lines.append(
        f"human subset: {HUMAN_SUBSET_SIZE} images, {human_labels[POSITIVE]} Positive, "
        f"{human_labels[NEGATIVE]} Negative "
        f"({100 * human_labels[POSITIVE] / HUMAN_SUBSET_SIZE:.1f}% Positive); "
        f"pneumonia subtypes {dict(human_sub)}"
    )

    hashes = {}
    for split in ("train", "val", "test"):
        hashes[split] = {sha256(p) for p, _ in splits[split]}
    lines.append("")
    lines.append(f"exact duplicate images, train & test: {len(hashes['train'] & hashes['test'])}")
    lines.append(f"exact duplicate images, train & val:  {len(hashes['train'] & hashes['val'])}")
    lines.append(f"exact duplicate images, val & test:   {len(hashes['val'] & hashes['test'])}")

    patients = {s: {patient_key(p.name) for p, _ in splits[s]} for s in splits}
    lines.append(f"patient IDs shared by train & test: {len(patients['train'] & patients['test'])}")
    lines.append(f"patient IDs shared by train & val:  {len(patients['train'] & patients['val'])}")
    lines.append(f"patient IDs shared by val & test:   {len(patients['val'] & patients['test'])}")

    report = "\n".join(lines) + "\n"
    out = PROJECT / "data" / "data_checks.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text(report)
    print(report)


if __name__ == "__main__":
    main()

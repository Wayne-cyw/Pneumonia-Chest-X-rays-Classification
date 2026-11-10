"""Copy the human-subset images, renamed to their Example ID (no label or file name), for labeller.html.

Run from the project folder:  python approaches/1_human_baseline/prepare_subset.py

The 40 IDs in subset_ids.csv are a fixed random sample of the test set (already labelled); they are
read here, never re-drawn. Image paths come from data/image_index.csv.
"""
import csv
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    ids = [r["Example ID"] for r in csv.DictReader(open(HERE / "subset_ids.csv"))]
    index = {r["Example ID"]: r for r in csv.DictReader(open(PROJECT / "data" / "image_index.csv"))}
    assert all(index[i]["split"] == "test" for i in ids), "the human subset must come from the test set"
    image_dir = HERE / "images"
    image_dir.mkdir(exist_ok=True)
    for i in ids:
        src = PROJECT / index[i]["path"]
        shutil.copyfile(src, image_dir / f"{i}{src.suffix.lower()}")
    positives = sum(int(index[i]["label"]) for i in ids)
    print(f"copied {len(ids)} images to {image_dir} ({positives} Positive, {len(ids) - positives} Negative)")

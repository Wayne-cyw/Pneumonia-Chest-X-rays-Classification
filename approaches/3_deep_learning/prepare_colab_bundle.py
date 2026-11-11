"""Build a small folder you can upload to Google Drive to run deep_learning.ipynb on Colab.

Run once from the project folder (needs numpy, pandas and pillow; the project .venv has them):
    python approaches/3_deep_learning/prepare_colab_bundle.py

It writes  local/colab_upload/csc211_project/  containing only what the notebook needs:
    data/image_index.csv, data/data_checks.txt
    eval/ground_truth.csv, eval/eval.py, eval/predictions/{human_eval,symbolic_eval}.csv
    approaches/1_human_baseline/subset_ids.csv
    approaches/3_deep_learning/deep_learning.ipynb
    dl_work/images_224.npy + dl_work/ids_224.json   (every X-ray as a 224x224 grayscale uint8 array, about 290 MB)

Upload the csc211_project folder to Drive (My Drive/csc211_project), open the notebook in Colab, and run it.
The image cache replaces the 1.2 GB raw dataset, which is not needed on Colab. It must be built with the same
image order and resize the notebook uses: the notebook checks the Example ID list and rebuilds if it differs.
"""
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "local" / "colab_upload" / "csc211_project"
IMG = 224  # must match PROFILES["standard"]["img"] in the notebook

COPY = [
    "data/image_index.csv",
    "data/data_checks.txt",
    "eval/ground_truth.csv",
    "eval/eval.py",
    "eval/predictions/human_eval.csv",
    "eval/predictions/symbolic_eval.csv",
    "approaches/1_human_baseline/subset_ids.csv",
    "approaches/3_deep_learning/deep_learning.ipynb",
]


def load_gray(rel):
    # identical to load_gray() in the notebook: grayscale, LANCZOS resize to a IMG x IMG square, uint8
    im = Image.open(PROJECT / rel).convert("L").resize((IMG, IMG), Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=np.uint8)


if __name__ == "__main__":
    index = pd.read_csv(PROJECT / "data" / "image_index.csv")
    # same row order as the notebook in full mode: train, then val, then test, each in file order
    df = pd.concat([index[index.split == s] for s in ("train", "val", "test")], ignore_index=True)
    if not (PROJECT / df.path.iloc[0]).exists():
        raise SystemExit(f"raw images not found, e.g. {PROJECT / df.path.iloc[0]}")

    OUT.mkdir(parents=True, exist_ok=True)
    for rel in COPY:
        (OUT / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / rel, OUT / rel)

    work = OUT / "dl_work"
    work.mkdir(exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(8) as pool:
        X = np.stack(list(pool.map(load_gray, df.path)))
    np.save(work / f"images_{IMG}.npy", X)
    (work / f"ids_{IMG}.json").write_text(json.dumps(df["Example ID"].tolist()))
    print(f"{len(X)} images -> {X.shape} uint8 in {time.time() - t0:.0f}s")
    print(f"bundle ready: {OUT}")
    print("Upload it to Google Drive as  My Drive/csc211_project  and open deep_learning.ipynb in Colab.")

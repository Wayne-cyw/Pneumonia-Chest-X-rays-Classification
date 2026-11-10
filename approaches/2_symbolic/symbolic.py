import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]

SIZE = 256
LUNG_BOXES = [(0.15, 0.42, 0.27, 0.74), (0.58, 0.85, 0.27, 0.74)]
CHEST_BOX = (0.20, 0.80, 0.20, 0.80)


def load_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return image


def crop_content(gray):
    """Remove the black borders so framing differences between scanners matter less."""
    cols = np.where(gray.mean(axis=0) > 20)[0]
    rows = np.where(gray.mean(axis=1) > 20)[0]
    if len(cols) < 20 or len(rows) < 20:
        return gray
    return gray[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def normalize(gray):
    """Resize and stretch brightness per image (2nd..98th percentile -> 0..1) to cancel exposure."""
    small = cv2.resize(crop_content(gray), (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
    lo, hi = np.percentile(small, (2, 98))
    return np.clip((small - lo) / max(hi - lo, 1.0), 0.0, 1.0)


def _box(arr, box):
    x0, x1, y0, y1 = box
    h, w = arr.shape
    return arr[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def extract_features(path):
    """Measurements from the lung regions; each one has a plain-language meaning."""
    img = normalize(load_gray(path))
    lungs = np.concatenate([_box(img, b).ravel() for b in LUNG_BOXES])

    detail = img - cv2.GaussianBlur(img, (0, 0), 8)
    lung_detail = np.concatenate([_box(detail, b).ravel() for b in LUNG_BOXES])

    edges = cv2.Canny((img * 255).astype(np.uint8), 40, 100)
    lung_edges = np.concatenate([_box(edges, b).ravel() for b in LUNG_BOXES]) > 0

    chest_mean = float(_box(img, CHEST_BOX).mean())
    return {
        "lung_mean": float(lungs.mean()),
        "lung_dark_frac": float((lungs < 0.35).mean()),
        "lung_p10": float(np.percentile(lungs, 10)),
        "lung_detail": float(lung_detail.std()),
        "lung_edges": float(lung_edges.mean()),
        "lung_vs_chest": float(lungs.mean() / max(chest_mean, 1e-6)),
    }


THRESHOLDS = {
    "lung_edges": (0.24, -1),
    "lung_detail": (0.08, -1),
    "lung_mean": (0.60, +1),
}
VOTES_NEEDED = 2


def votes_for(features):
    """Which features point to pneumonia; shows why a prediction was made."""
    return {name: (features[name] - threshold) * sign > 0 for name, (threshold, sign) in THRESHOLDS.items()}


def predict(path):
    """Return "Positive" (pneumonia) if at least VOTES_NEEDED of the 3 features point to pneumonia."""
    votes = votes_for(extract_features(path))
    return "Positive" if sum(votes.values()) >= VOTES_NEEDED else "Negative"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sample = Path(sys.argv[1])
    else:
        sample = sorted((PROJECT / "data" / "sample").glob("*.jp*g"))[0]
    print(f"{sample.name}: {predict(sample)}")

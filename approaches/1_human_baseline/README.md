# Approach 1: Human baseline

**Question:** how well does a motivated person with no radiology training classify these X-rays? This gives the machine approaches a reference point. Context: [root README](../../README.md#51-approach-1-human-baseline).

## Files

| File | Purpose |
|---|---|
| `subset_ids.csv` | The 40 test-image IDs that were labelled (a fixed random sample of the 624 test images, seed 211). Never re-drawn. |
| `prepare_subset.py` | Copies those 40 images into `images/`, renamed to their Example ID, so no original file name (which contains "bacteria" or "virus") is visible. Asserts that every ID belongs to the test split. |
| `images/` | The 40 renamed images, e.g. `ex_023.jpeg`. |
| `labeller.html` | A self-contained browser tool: shows one image at a time; keys `P` (Positive) and `N` (Negative) label it, the arrow keys move; progress is saved in the browser; **Export** downloads a CSV. |
| `../../eval/predictions/human_eval.csv` | The recorded answers (`Example ID, Human Prediction, Notes`). |
| `../../eval/score_human_baseline.py` | Prints the human's accuracy; the full comparison is `eval/eval.py`. |

## How the labelling was done

- One labeller: the author, with no radiology training.
- Input identical to the machines': the image only (renamed to the example ID, scaled to fit the browser window). No file name, age, history or label.
- No ground truth was consulted before the labels were final. No timer and no notes.
- 28 of the 40 images are pneumonia and 12 are normal (70.0% pneumonia, against 62.5% in the full test set).

## Result

| Metric | Value |
|---|---|
| Correct | 27 of 40 (accuracy 0.675) |
| Recall | 0.643 (18 caught, 10 missed) |
| Specificity | 0.750 (9 cleared, 3 false alarms) |
| Precision | 0.857 |
| F1 | 0.735 |

On the same 40 images the symbolic rule also gets 27 correct and the neural network 39. The human and the rule disagree on 20 images, each right on 10, so the 40-image subset cannot say which of those two is better.

## Reproduce

```bash
python approaches/1_human_baseline/prepare_subset.py     # builds images/
open approaches/1_human_baseline/labeller.html           # label all 40 and press Export
# save the exported file as eval/predictions/human_eval.csv
python eval/score_human_baseline.py
```

**Limits.** n = 40 means one image is worth 2.5 accuracy points; one labeller, one session; the labeller is not a clinician.

Design reasoning for the subset and the protocol: [decision A8](../../docs/design-decisions.md#a8-a-fixed-random-sample-of-40-test-images-for-the-human).

# Design decisions

This document records the main choices made in the project: **what was decided, what the alternatives were, why this option was picked, what evidence supports it, and what would change the decision.** It is meant to be read next to the [root README](../README.md), which explains the project itself.

Each entry uses the same fields:

- **Decision**: what was chosen.
- **Alternatives**: what else was reasonable.
- **Why**: the reasoning.
- **Evidence**: the file or figure in the repository that backs it up.
- **Consequence / would change if**: what it costs, and what result would make me revisit it.

All file paths are relative to the repository root. Numbers marked "val" are validation numbers; "test" numbers were measured once at the end.

## Summary

| # | Area | Decision |
|---|---|---|
| A1 | Data | One project-wide train/validation/test split, stored in one file |
| A2 | Data | Split by patient where a patient ID exists |
| A3 | Data | Test set = the dataset's own test folder, unchanged |
| A4 | Data | Example IDs are shuffled; file names are never an input |
| A5 | Evaluation | F1 is the primary metric, always shown beside specificity, balanced accuracy and MCC |
| A6 | Evaluation | The test set is used once; the code enforces it |
| A7 | Evaluation | Confidence intervals come from a patient-level bootstrap |
| A8 | Evaluation | A fixed random sample of 40 test images for the human baseline |
| B1 | Symbolic | Hand-written rule, no training on labels |
| B2 | Symbolic | Normalise, crop, then measure fixed lung boxes |
| B3 | Symbolic | Three features: edge density, fine detail, brightness |
| B4 | Symbolic | Thresholds at the midpoint of the class medians |
| B5 | Symbolic | Two of three votes decide |
| C1 | Deep learning | Keras, as the project specification requires |
| C2 | Deep learning | Transfer learning from an ImageNet backbone |
| C3 | Deep learning | Grayscale 224×224 input, cached as uint8, expanded to 3 channels inside the model |
| C4 | Deep learning | One sigmoid output with binary cross-entropy |
| C5 | Deep learning | Class weights for the imbalance |
| C6 | Deep learning | Augmentation inside the model, without horizontal flips |
| C7 | Deep learning | Two-phase training, BatchNorm frozen, AdamW with cosine decay |
| C8 | Deep learning | Staged tuning, not a full grid |
| C9 | Deep learning | Choose by validation AUROC; differences under 0.01 go to the simpler model |
| C10 | Deep learning | Long run, no early stopping, two seeds, keep the best-AUROC epoch |
| C11 | Deep learning | Decision threshold by Youden's J on validation |
| C12 | Deep learning | A tiny scratch CNN as a labelled reference, never a candidate |
| C13 | Deep learning | Grad-CAM, bacterial-vs-viral analysis and a cost table for the "three considerations" |
| C14 | Deep learning | Run on Colab from a prebuilt image cache |
| D | Outcomes | Decisions that did not work as hoped |

---

## A. Data and evaluation

### A1. One project-wide split, defined in one place

- **Decision.** The train/validation/test assignment of every image is a column (`split`) in `data/image_index.csv`. Every approach reads it; no approach splits the data itself.
- **Alternatives.** Let each approach make its own split (the first version did this for the symbolic rule).
- **Why.** The four approaches are only comparable if they are trained, tuned and scored on the same images. A per-approach split also invites accidents, such as an approach tuned on images another approach used as test.
- **Evidence.** The refactor commit `b5cd937`; `data/prepare_data.py`. During the refactor the test IDs, `eval/ground_truth.csv` and the existing predictions were required to stay byte-identical, so earlier results stayed valid.
- **Consequence / would change if.** Train/validation sizes changed slightly (4,190 / 1,026 before; 4,185 / 1,047 now), so the earlier written report in `docs/` no longer matches the repository to the last digit.

### A2. Split by patient where a patient ID exists

- **Decision.** Pneumonia images are grouped by the patient number in the file name (`person<N>_...`; the bacterial and viral files with the same N are one patient). The 80/20 train/validation split uses `StratifiedGroupKFold` (5 folds, fold 0), which keeps groups together and keeps the class ratio equal in both parts. The code asserts no patient is in both.
- **Alternatives.** A random image-level split; a split that ignores class balance.
- **Why.** One child contributes several X-rays. An image-level split lets a model recognise the *child* rather than the *disease*, which inflates validation scores and misleads every tuning decision made with them.
- **Evidence.** `data/prepare_data.py` (the `assert`); `data/data_checks.txt` shows both parts at 74.2% pneumonia and zero patient overlap.
- **Consequence / would change if.** Normal files carry no usable patient ID (the `IM-xxxx` number in their names is a coincidence), so each normal image is its own group. One child's normal images could therefore sit on both sides, making validation slightly optimistic. A real patient ID for normals would fix this.

### A3. The test set is the dataset's own test folder, unchanged

- **Decision.** `split == "test"` is exactly Kaggle's `test/` folder (624 images). Kaggle's 16-image `val/` folder is merged into the train/validation pool.
- **Alternatives.** Re-split everything into new train/validation/test sets; keep the 16-image folder as validation.
- **Why.** The Kaggle test folder is a published, fixed benchmark, so results are comparable with other work. Sixteen images (8 normal, 8 pneumonia) are far too few to tune on.
- **Evidence.** `data/data_checks.txt`: 170 pneumonia patient numbers appear in both Kaggle's train and test folders, and no exact duplicate images exist between folders.
- **Consequence / would change if.** The test set differs from train/validation (62.5% versus about 74% pneumonia, and, as the results suggest, its normal images behave differently; see [results](results-and-limitations.md#4-why-validation-was-too-optimistic)). The possible patient overlap cannot be resolved from file names, so pneumonia test scores may be a little optimistic. This is reported, not fixed.

### A4. Shuffled IDs; file names never an input

- **Decision.** Images get IDs like `ex_001` after a seeded shuffle (seed 211), so the ID order carries no class information. No approach receives a file name or path.
- **Alternatives.** Use original file names or sorted IDs.
- **Why.** Pneumonia file names contain "bacteria" or "virus", which gives the answer away. Sorted IDs would put all normal images first.
- **Evidence.** `data/prepare_data.py`; the human labeller shows only the ID-named image. The subtype is read from file names only after prediction, for the fairness analysis.
- **Consequence / would change if.** The shuffle order must never change, because every ID in `eval/` depends on it; the code comments say so.

### A5. F1 as the primary metric, never alone

- **Decision.** The primary metric is **F1 with PNEUMONIA as the positive class**. Recall, specificity, precision, accuracy and balanced accuracy are always printed beside it, and the neural network is also scored with MCC, AUROC, AUPRC, Brier score and calibration error.
- **Alternatives.** Accuracy; recall only; AUROC only.
- **Why.** Accuracy rewards guessing the majority class (62.5% for "always pneumonia"). Recall alone also rewards always answering "pneumonia". AUROC cannot be computed for the human or the rule, which only give yes/no answers, so it cannot be the headline metric for a four-way comparison.
- **Evidence.** The "always positive" row in [`approaches/3_deep_learning/results/test_metrics.csv`](../approaches/3_deep_learning/results/test_metrics.csv): F1 0.769, **higher than the symbolic rule's 0.763**, but specificity 0.000 and MCC 0.000.
- **Consequence / would change if.** F1 ignores true negatives, so a method that over-calls pneumonia still looks decent. That is exactly the weakness of the deep model on the test set, and it is why the specificity column matters.

### A6. The test set is used once, and the code enforces it

- **Decision.** The rule's test run requires `--confirm`; the notebook's test cell refuses to run again if `results/test_scores.csv` exists unless `ALLOW_SECOND_TEST_RUN` is set. Nothing tuned anything on the test set.
- **Alternatives.** Rely on discipline.
- **Why.** Every look at the test set while choosing settings leaks information into the choice and makes the final score optimistic. A code guard makes an accidental second look impossible.
- **Evidence.** `approaches/2_symbolic/make_symbolic_eval.py`; the notebook section "Run the test set ONCE".
- **Consequence / would change if.** It had a real cost here: after the test run showed the threshold did not transfer, I could not re-tune it on the test set. A threshold picked with the test labels would reach specificity about 0.88 and recall about 0.92, but such a number would not be a valid test result, so it is mentioned only as a diagnosis.

### A7. Patient-level bootstrap for confidence intervals

- **Decision.** 95% intervals come from resampling patients (each normal image is its own group) with replacement.
- **Alternatives.** Resample images; report no intervals.
- **Why.** Images from one child are not independent, so resampling single images makes intervals too narrow. Intervals show which gaps between approaches are real.
- **Evidence.** The `_ci_lo` and `_ci_hi` columns of `test_metrics.csv`; the notebook section "Metrics with confidence intervals". Deep learning versus the rule: recall, F1 and MCC intervals do not overlap; specificity intervals do.
- **Consequence / would change if.** Normal images are treated as independent, which slightly narrows their part of the interval.

### A8. A fixed random sample of 40 test images for the human

- **Decision.** The author labelled a random 40 of the 624 test images (seed 211; 28 pneumonia, 12 normal), frozen in `approaches/1_human_baseline/subset_ids.csv`.
- **Alternatives.** Label all 624 (impractical); label a hand-picked "typical" set (biased).
- **Why.** A random sample is unbiased; 40 images is a workload one person can do without fatigue changing the answers.
- **Evidence.** `eval/eval.py` prints the class balance of the subset (70.0% pneumonia against 62.5% overall).
- **Consequence / would change if.** With n = 40 one image is 2.5 accuracy points, and the subset turned out easier than the full test set for the network (accuracy 0.975 versus 0.877), so subset comparisons are indicative only. More labelled images, or a second labeller, would narrow this.

---

## B. Symbolic (hand-written rule)

### B1. A rule written by hand, with no model fitted to labels

- **Decision.** The approach is a fixed procedure with three thresholds; no classifier is trained.
- **Alternatives.** Train logistic regression or a random forest on the same measurements.
- **Why.** This approach is defined as hand-written logic, and a classifier fitted to labels belongs in the deep learning/machine learning approach. A rule is also fully explainable: for any image one can print which measurement voted which way.
- **Evidence.** `approaches/2_symbolic/symbolic.py` (`votes_for` returns the reason for each answer).
- **Consequence / would change if.** A fitted classifier on the same features would likely do a little better; the project deliberately does not.

### B2. Normalise, then measure fixed lung boxes

- **Decision.** Crop black borders, resize to 256×256, stretch the 2nd to 98th brightness percentile to 0 to 1, then measure only inside two fixed boxes (x 15–42% and 58–85%, y 27–74%).
- **Alternatives.** Measure the whole image; segment the lungs automatically.
- **Why.** Scans differ in framing and exposure; the stretch and crop remove much of that. Restricting to the lungs keeps shoulders, arms and the abdomen out of the measurement.
- **Evidence.** `symbolic.py` (`normalize`, `LUNG_BOXES`).
- **Consequence / would change if.** Fixed boxes cannot adapt to a child who is rotated or off-centre, and a whole-lung average misses small patchy pneumonia. The error analysis in the written report in `docs/` found many missed cases of exactly that kind. Lung segmentation would address it.

### B3. Three features: edge density, fine detail, brightness

- **Decision.** Of six candidate measurements, the rule uses edge density (`lung_edges`), fine detail (`lung_detail`) and mean brightness (`lung_mean`).
- **Alternatives.** The other three candidates: share of dark pixels, 10th-percentile brightness, lung-to-chest brightness ratio.
- **Why.** Each was measured on the *training* images and ranked by how well it alone separates the classes (AUROC; a value below 0.5 just means the direction is flipped). The chosen three are the strongest and each has a plain-language reason (hazy lungs: fewer sharp edges, less fine contrast, whiter).
- **Evidence.** `python approaches/2_symbolic/explore_features.py`, train images:

| Feature | Single-feature AUROC (direction-corrected) | Used? |
|---|---|---|
| `lung_edges` | 0.911 (shown as 0.089 before flipping) | yes |
| `lung_detail` | 0.829 (0.171) | yes |
| `lung_mean` | 0.825 | yes |
| `lung_dark_frac` | 0.741 (0.259) | no |
| `lung_p10` | 0.744 | no |
| `lung_vs_chest` | 0.720 | no |

  The values on the validation images are within 0.03 of these.
- **Consequence / would change if.** The three chosen features are correlated (all respond to haze), so a vote of three is less independent than it looks.

### B4. Thresholds at the midpoint of the class medians

- **Decision.** Each cut-off is halfway between the NORMAL median and the PNEUMONIA median on the training images: edges 0.24, detail 0.08, brightness 0.60.
- **Alternatives.** The cut that maximises balanced accuracy on the training images.
- **Why.** The two methods agree almost exactly (brightness 0.60 versus 0.61, the others identical), so the simpler rule, easy to state by hand, wins.
- **Evidence.** `python approaches/2_symbolic/tune_rule.py` prints both threshold sets; the vote results differ by about 0.01 in balanced accuracy.
- **Consequence / would change if.** Thresholds are rounded to two decimals, as in `symbolic.py`.

### B5. Two of three votes decide

- **Decision.** The rule answers PNEUMONIA when at least 2 of the 3 features vote for it.
- **Alternatives.** 1 of 3 (answer pneumonia easily) or 3 of 3 (only when all agree).
- **Why.** A single required vote catches more pneumonia but clears few normals; three votes clear normals but miss half the pneumonia. Two is the balance. The choice was made on train and validation only.
- **Evidence.** `tune_rule.py`, midpoint thresholds:

| Votes needed | Balanced accuracy, train | Balanced accuracy, validation | Recall (train) | Specificity (train) |
|---|---|---|---|---|
| 1 of 3 | 0.797 | 0.783 | 0.894 | 0.700 |
| **2 of 3** | **0.824** | **0.798** | 0.745 | 0.903 |
| 3 of 3 | 0.756 | 0.720 | 0.538 | 0.973 |

- **Consequence / would change if.** On the test set the rule's recall fell to 0.697 and specificity to 0.782. The gap between train and test is a hint that the test images are different, not that the rule was tuned on them.

---

## C. Deep learning

### C1. Keras

- **Decision.** The training and inference code is Python with Keras on a TensorFlow backend.
- **Alternatives.** PyTorch.
- **Why.** The project specification requires Keras; PyTorch was out of scope.
- **Evidence.** The notebook header; run info `results/run_info.json` (TensorFlow 2.21.0, Keras 3.13.2).
- **Consequence / would change if.** A switch to PyTorch would be a moderate rewrite.

### C2. Transfer learning from an ImageNet backbone

- **Decision.** Use a pretrained network from `keras.applications` and adapt it.
- **Alternatives.** Train a network from scratch.
- **Why.** About 4,000 training images are too few to learn good image features from nothing, and transfer learning is the standard choice when a pretrained model exists.
- **Evidence.** The learning curve (`results/learning_curve.csv`, figure 22): at 10% of the training data pretrained reaches validation AUROC 0.969 against 0.737 for a scratch CNN; at 25% 0.982 against 0.946.
- **Consequence / would change if.** At 50% and 100% of the data the scratch CNN matches or beats the pretrained model on validation (0.9905 against 0.9885 at 100%). So the benefit here is data efficiency, not final accuracy. ImageNet contains photographs, not X-rays, which is a limitation to state.

### C3. Grayscale 224×224 input, cached, expanded to 3 channels in the model

- **Decision.** Each X-ray is converted to grayscale, resized once to 224×224 and stored as `uint8`. Inside the model a frozen 1×1 convolution with all-ones weights copies it into 3 channels.
- **Alternatives.** Decode the original JPEGs every epoch; use a `Concatenate` layer to make 3 channels.
- **Why.** Decoding 1,500-pixel JPEGs every epoch would dominate training time. The frozen convolution keeps the model a single-input network, which Grad-CAM needs.
- **Evidence.** Found while probing Keras: `Concatenate` made Grad-CAM fail (it then needs 3 inputs).
- **Consequence / would change if.** Resizing to a square distorts the aspect ratio slightly for every image; the same transform is used for every split.

### C4. One sigmoid output with binary cross-entropy

- **Decision.** The head is `Dropout, Dense(1, sigmoid)` with binary cross-entropy loss.
- **Alternatives.** A 2-unit softmax with categorical loss.
- **Why.** They are mathematically equivalent for two classes. A single sigmoid gives one number, P(pneumonia), which is exactly what the predictions file, the AUROC and the threshold need. It is the standard two-class layout.
- **Evidence.** Notebook section 2.
- **Consequence / would change if.** Switching to a 2-unit softmax head would be a small change.

### C5. Class weights for the imbalance

- **Decision.** Training uses `class_weight` so that normal images count about 2.9 times as much as pneumonia images (train is 74% pneumonia).
- **Alternatives.** Resample the data; no correction.
- **Why.** Without it the cheapest way to lower the loss is to predict pneumonia more often.
- **Evidence.** Ablation `results/ablation.csv`: removing class weights changed validation AUROC from 0.9885 to 0.9886, i.e. no measurable effect on ranking.
- **Consequence / would change if.** AUROC ignores how probabilities are shifted, so class weights mainly act on the probabilities (and the right threshold), not on the ranking. That interacts with C11.

### C6. Augmentation inside the model, without horizontal flips

- **Decision.** Mild random rotation, zoom, translation, brightness and contrast, implemented as Keras layers that switch off at prediction time. No horizontal flip.
- **Alternatives.** Offline augmented copies; strong augmentation; flips.
- **Why.** The heart is on the left of the chest; a flipped image shows anatomy that does not occur.
- **Evidence.** Ablation: adding flips gave validation AUROC 0.9882 against 0.9885 (no gain). "No augmentation" scored 0.9932, so on validation augmentation did not help either.
- **Consequence / would change if.** I kept "mild" because augmentation is standard practice and costs nothing at prediction time, but the evidence does not show that it helps; the test set differs from validation, so validation is not the right judge of augmentation.

### C7. Two-phase training, BatchNorm frozen, AdamW with cosine decay

- **Decision.** Phase A trains only the new head (learning rate 1e-3, 3 epochs). Phase B continues with a smaller rate and cosine decay; BatchNorm layers stay frozen; the optimiser is AdamW.
- **Alternatives.** Train everything from the start; unfreeze BatchNorm.
- **Why.** A randomly initialised head can send large gradients into pretrained layers and damage them; BatchNorm statistics should not drift on a small, different dataset (standard Keras fine-tuning advice).
- **Evidence.** Notebook section 4. A bug caught here: the augmentation block is itself a `keras.Model`, so a "find the backbone" helper must skip it, otherwise freezing silently does nothing. The notebook now asserts that the trainable-parameter count changes with the unfreeze level; the saved `stage1_capacity.csv` shows 1,281 / 2.7 M / 4.6 M / 5.8 M trainable parameters.

### C8. Staged tuning, not a full grid

- **Decision.** Each stage fixes one decision and passes the winner on: (1) how much of the backbone to unfreeze, (2) which backbone, (3) learning rate, weight decay, dropout and augmentation by random search, (4) a long run.
- **Alternatives.** A full grid over everything (hundreds of runs).
- **Why.** A grid would need hundreds of runs; random search covers more values of every setting for the same budget, and the staged design keeps each result interpretable.
- **Evidence.** `results/stage1_capacity.csv`, `stage2_backbones.csv`, `stage3_random_search.csv`; `results/training_minutes_by_stage.csv` (about 61 minutes in total).
- **Consequence / would change if.** Stages interact (the best learning rate may depend on the backbone), and a staged search can miss that.

### C9. Choose by validation AUROC; ties within 0.01 go to the simpler model

- **Decision.** Candidates are ranked by validation AUROC. Because validation has about 1,000 images, differences below 0.01 are treated as noise and the smaller or more frozen model wins.
- **Alternatives.** Always take the highest score; rank by F1 at a fixed threshold.
- **Why.** AUROC does not depend on a threshold, so it is a fair way to compare models before a threshold exists. The tolerance stops tuning noise from picking a bigger model; this is common advice for small validation sets.
- **Evidence.** The three tuning tables:

| Stage | Option | Validation AUROC |
|---|---|---|
| 1. Unfreeze (EfficientNetV2B0) | 0% (head only) | 0.9908 |
| | 25% | 0.9975 |
| | 50% | 0.9983 |
| | 100% | 0.9992 |
| 2. Backbone (head only) | MobileNetV2 (2.26 M parameters, 9.6 MB) | 0.9882 |
| | EfficientNetV2B0 (5.92 M, 24.7 MB) | 0.9908 |
| | ConvNeXtTiny (27.8 M, 112 MB) | 0.9877 |
| 3. Random search (5 trials plus defaults) | best (lr 9.1e-5, wd 9.2e-4, dropout 0, mild aug) | 0.9880 |
| | worst (lr 3.2e-4, dropout 0.4, strong aug) | 0.9843 |

- **Consequence / would change if.** The rule produced a head-only MobileNetV2 with 1,281 trainable numbers. In stage 1 the score rose steadily with more unfreezing (0.9908 to 0.9992), a total spread of 0.008, just inside the tolerance, so the monotone trend was overruled by the tie rule. Whether more fine-tuning would have transferred better to the shifted test set is unknown, and cannot be tested now without breaking A6. Stage 3 was essentially flat (0.984 to 0.988): the model was not sensitive to those settings.

### C10. Long run, no early stopping, two seeds, keep the best-AUROC epoch

- **Decision.** The winning configuration is trained for 40 epochs after 3 head-only epochs, twice (seeds 211 and 212), with no early stopping. The weights of the epoch with the best validation AUROC are kept; the seed with the better validation AUROC (212, epoch 35, 0.9923) is the final model.
- **Alternatives.** Early stopping; a single seed.
- **Why.** Training past the best epoch shows what longer training does; two seeds show run-to-run noise (0.9903 versus 0.9923).
- **Evidence.** `results/long_training_runs.csv`; figure 06: validation AUROC flattens after about 12 epochs, and the best epochs (35 and 39 of 43) are late but the gain after epoch 12 is small.
- **Consequence / would change if.** Two seeds is a small sample for estimating noise. Choosing between seeds on validation is legitimate but slightly optimistic for validation.

### C11. Threshold by Youden's J on validation

- **Decision.** The cut-off that maximises recall + specificity − 1 on validation, which came out at **0.738**, is frozen before the test run.
- **Alternatives.** 0.5; maximise F1 on validation; require recall at least 0.95 and then take the best specificity.
- **Why.** Youden's J does not depend on how common pneumonia is, and the prevalence differs between validation (about 74%) and test (62.5%). Maximising F1 would favour recall at 74% prevalence. 0.5 is not neutral because the loss is class-weighted.
- **Evidence.** `results/run_info.json`; figure 15 (threshold sweep).
- **Consequence / would change if.** This did not transfer. At 0.738 validation specificity is about 0.97 (read from figure 15) but test specificity is 0.709 ([results](results-and-limitations.md#4-why-validation-was-too-optimistic)). A prevalence-independent rule fixes only the prevalence part of the shift, not the images themselves being different. This is the most important decision that did not work out.

### C12. A tiny scratch CNN, only as a labelled reference

- **Decision.** A four-block CNN (98,292 parameters, 0.49 MB) trained from scratch appears in the capacity, learning-curve and cost figures. It is never a candidate model and never touched the test set.
- **Alternatives.** Leave it out; make it a candidate.
- **Why.** It shows what pretraining buys and gives a lower bound on size and speed. The project's design is transfer learning, so a from-scratch model is kept as a reference only.
- **Evidence.** `results/cost_and_capacity.csv`: val AUROC 0.9927 at about 5 ms per image (batched), higher than the three head-only pretrained backbones in that table, though lower than the fine-tuned EfficientNetV2B0 runs of stage 1 (up to 0.9992).
- **Consequence / would change if.** Because it was never run on the test set, I cannot say whether it would have generalised better or worse than the pretrained model, given the validation/test shift.

### C13. The "three considerations": explainability, fairness, cost

- **Decision.** Explainability: Grad-CAM heat maps for correct and incorrect examples. Fairness: correct-rate for normal, bacterial and viral subgroups. Cost: parameters, size on disk, training seconds per epoch and minutes per stage.
- **Alternatives.** Privacy instead of cost.
- **Why.** Each one needs evidence from the experiments, not just discussion. Subtype is the only subgroup readable from the data.
- **Evidence.** Figures 17, 19, 21 and `results/subgroup_performance.csv`, `results/cost_and_capacity.csv`.
- **Consequence / would change if.** Age, sex and scanner are not in the data, so other fairness checks are impossible. Grad-CAM shows where the model looked, not why. The inference-latency columns are unreliable (warm-up included); only size, parameters and seconds per epoch are used.

### C14. Colab, from a prebuilt image cache

- **Decision.** Training ran on a Colab GPU. `prepare_colab_bundle.py` builds a folder of about 290 MB with every image already resized to 224×224, instead of uploading the 1.2 GB raw dataset.
- **Alternatives.** Train locally on CPU; upload the raw data.
- **Why.** Speed and upload size. Every experiment is cached on Drive so a disconnect resumes.
- **Evidence.** `prepare_colab_bundle.py`; `results/run_info.json` (a GPU device; TensorFlow 2.21.0, Keras 3.13.2 on Colab, while a local environment had Keras 3.15).
- **Consequence / would change if.** Software versions differ between Colab and a local machine; a model saved on one should be reloaded and tested on the other before relying on it.

---

## D. Decisions that did not work as hoped

These are recorded because understanding why something failed is part of the project.

| Decision | What happened | What it taught |
|---|---|---|
| Threshold from validation (C11) | Test specificity 0.709 instead of about 0.97 | Tuning on data from the same pool as training gives optimistic operating points; the test folder is different. |
| "No pretraining" ablation | Validation AUROC exactly 0.500 | BatchNorm stayed frozen at its start-up statistics, and even when trained its momentum of 0.999 averages over about 1,000 steps, far more than a short run provides. The network output one constant. The notebook now trains BatchNorm with momentum 0.9 for that variant, but the saved results were not regenerated; ignore that row until a Colab re-run. |
| "Frozen backbone" ablation | Identical to the full recipe (0.9885) | The final recipe is already head-only, so this row was a duplicate. The notebook now runs "All layers unfrozen" instead; pending a Colab re-run. |
| Train versus validation loss figures | Loss panels exaggerate the overfitting gap | Training loss was class-weighted but validation loss was not. The notebook now weights both; the saved figures 06 and 07 pre-date that fix. |
| Cost timings | Single-image latency of 2.4 s for ConvNeXtTiny | Almost certainly warm-up overhead. The notebook's timing code now warms up first, but the saved file is from the older code; only batch timings and sizes are trusted. |
| Pretrained beats scratch | It does not at full data | The honest claim is "data-efficient", not "more accurate". |

# Bat calls

Deep Learning class project (team of 3). A model listens to a call of an Egyptian fruit bat and predicts what the bats were doing: Biting, Feeding, Fighting, Mating protest or Sleeping.

The student driving this repo is new to deep learning and must be able to explain every line in class.

## How to work here

- Keep everything as simple as possible: three short scripts, no comments or docstrings in the code, no extra files or folders.
- One problem at a time. After a run, say in plain language what the problem is, offer 2-3 fixes with their trade-offs and a recommendation, then stop. **The student chooses the fix.** Don't apply fixes that weren't chosen, and don't combine several changes in one step.
- Explain in the chat, not in the code. Never fake results; say when a difference is too small to mean anything.
- Compare fixes on the cross-validation score only (one run of `train.py` takes about 17 minutes). The test days are used once, at the very end.
- Commit each chosen fix separately and push, then add a row to the log below.

## The pieces

- `download.py`: gets the labels and every recording of the 5 contexts from Figshare (Prat et al. 2017, https://doi.org/10.1038/sdata.2017.143, CC BY 4.0) into `data/audio/` (50,998 recordings, about 54 GB).
- `prepare.py`: first 2 seconds of each annotated call become a 128x256 spectrogram (0-62.5 kHz). The last 10% of recording days are the test set (fold -1). The other days are divided into 5 folds (0-4) with `StratifiedGroupKFold`: a day is never split, and every fold has a similar mix of contexts. Then Sleeping calls are removed at random (fixed seed) until there are as many as Fighting calls: 7,963 of 33,997 are kept, everywhere including the test days. That leaves 26,735 calls: 23,526 in the folds (about 4,700 each) and 3,209 test calls. Writes `data/x.npy` and `data/y.csv`.
- `train.py`: 5-fold cross-validation. For each fold a fresh ResNet18 (pretrained on ImageNet, all layers trained) is trained on the other 4 folds and checked on the held-out fold. In training, one random frequency band and one random time band (each up to 16 pixels) of every picture are set to silence; held-out calls are never blanked. The model gives its two most likely contexts per call. Prints train and validation accuracy per epoch, a per-context report over all 5 held-out folds, and the accuracy with one guess and within two guesses next to the blind-guess numbers (always answering the most common context, or the two most common). `train.py test` skips the cross-validation: it trains one model on all 5 folds and checks it on the test days.

The folds mix days from all months, so a held-out day has neighbouring days in training. The cross-validation score is therefore higher than a score on later days (the test set) will be. It is for comparing fixes, not a prediction of the test score.

Run with `.venv\Scripts\python <script>`. `data/` is not in git.

## Log: problems and chosen fixes

| # | Problem | Fix chosen | Validation result |
|---|---|---|---|
| 0 | First version, nothing fixed yet (plain training, 5 epochs, one validation block = days 80-90%) | none | accuracy 0.758, macro-F1 0.440 after epoch 5; best epoch was 3 (accuracy 0.790); train accuracy reached 0.938. Always answering "Sleeping" scores 0.699. Recall: Biting 0.05, Feeding 0.70, Fighting 0.37, Mating protest 0.12, Sleeping 0.90 |
| 1 | The student asked for cross-validation instead of one validation block | 5-fold cross-validation by recording day | accuracy 0.801, macro-F1 0.580 after epoch 5 (always "Sleeping": 0.640). Mean validation accuracy per epoch: 0.804, 0.808, 0.812, 0.802, 0.802; train accuracy reaches about 0.95. Recall: Biting 0.04, Feeding 0.76, Fighting 0.64, Mating protest 0.44, Sleeping 0.91 |
| 2 | Too few non-Sleeping calls (student's idea: add a copy of every non-Sleeping training call rotated by 90 degrees, squeezed back to 128x256) | tested with a separate copy of `train.py`; dropped by the student, never in the code | accuracy 0.785, macro-F1 0.563 after epoch 5; mean validation accuracy per epoch 0.799, 0.812, 0.811, 0.806, 0.785. Recall: Biting 0.02, Feeding 0.85, Fighting 0.61, Mating protest 0.54, Sleeping 0.87. No gain over row 1 |
| 3 | Blind guessing ("always Sleeping") scored 0.640 because Sleeping was 64% of the calls | student's choice: keep only as many Sleeping calls as Fighting calls (random, also in the test set) | accuracy 0.697 against a blind guess of 0.317; macro-F1 0.604 after epoch 5. Recall: Biting 0.18, Feeding 0.82, Fighting 0.73, Mating protest 0.55, Sleeping 0.71 (mean recall 0.596, was 0.557 in row 1). Accuracy is not comparable with rows 0-2 because the set of calls changed |
| 4 | One guess per call hides how close the model is | student's choice: also score "right context within the model's two guesses" | within two guesses 0.885 against a blind two-guess score of 0.611. Per context: Biting 0.68, Feeding 0.92, Fighting 0.89, Mating protest 0.69, Sleeping 0.94 |
| 5 | The model memorizes: validation accuracy was best at epochs 1-2 and fell to 0.697 by epoch 5 while train accuracy reached 0.93 | student's choice: blank a random frequency band and time band of every training picture, still 5 epochs | one guess 0.720 (blind 0.317), within two guesses 0.905 (blind 0.611), macro-F1 0.619 after epoch 5. Mean validation accuracy per epoch: 0.704, 0.716, 0.728, 0.730, 0.720 (two guesses: 0.892, 0.902, 0.905, 0.905, 0.905); train accuracy 0.74 rising to 0.85. Recall: Biting 0.15, Feeding 0.86, Fighting 0.78, Mating protest 0.56, Sleeping 0.70. Within two guesses: Biting 0.59, Feeding 0.93, Fighting 0.94, Mating protest 0.73, Sleeping 0.95 |
| 6 | The student asked for the score on the test days (first look at them, 2026-11-13) | `train.py test`: one model trained on all 5 folds for 5 epochs, checked on the 3,209 test calls | TEST DAYS, not validation: one guess 0.607 (blind 0.326), within two guesses 0.818 (blind 0.568), macro-F1 0.509; train accuracy 0.848. Recall: Biting 0.005 (2 of 403), Feeding 0.91, Fighting 0.50, Mating protest 0.61, Sleeping 0.66. Within two guesses: Biting 0.47, Feeding 0.94, Fighting 0.75, Mating protest 0.80, Sleeping 0.90. Test accuracy per epoch: 0.541, 0.577, 0.572, 0.613, 0.607. Cross-validation had said 0.720 and 0.905. The test days hold more Biting (12.6% against 5.9% in the folds) and Mating protest (15.1% against 7.9%) and less Fighting (15.5% against 31.7%); the cross-validation recalls applied to the test mix would give 0.659, so about half of the drop is the different mix and half is the model doing worse on later days (mostly Fighting and Biting) |

Rows 3 and 4 come from one run. Mean validation accuracy per epoch in that run: 0.721, 0.721, 0.710, 0.709, 0.697 (within two guesses: 0.900, 0.904, 0.896, 0.886, 0.885); train accuracy reaches about 0.93.

Still open: the cross-validation score (0.720) is 11 points higher than the test score (0.607), so it does not show whether a fix helps on later days. Biting is almost never the first guess (15% of Biting calls in cross-validation, 2 of 403 on the test days). The number of epochs has not been chosen by the student. The test days have now been looked at once; every further look makes the test score less trustworthy, so fixes are still compared on cross-validation and the test days are checked again only at the end.

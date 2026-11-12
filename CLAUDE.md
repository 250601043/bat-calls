# Bat calls

Deep Learning class project (team of 3). A model listens to a call of an Egyptian fruit bat and predicts what the bats were doing: Biting, Feeding, Fighting, Mating protest or Sleeping.

The student driving this repo is new to deep learning and must be able to explain every line in class.

## How to work here

- Keep everything as simple as possible: three short scripts, no comments or docstrings in the code, no extra files or folders.
- One problem at a time. After a run, say in plain language what the problem is, offer 2-3 fixes with their trade-offs and a recommendation, then stop. **The student chooses the fix.** Don't apply fixes that weren't chosen, and don't combine several changes in one step.
- Explain in the chat, not in the code. Never fake results; say when a difference is too small to mean anything.
- Compare fixes on the cross-validation score only (one run of `train.py` takes about 30 minutes). The test days are used once, at the very end.
- Commit each chosen fix separately and push, then add a row to the log below.

## The pieces

- `download.py`: gets the labels and every recording of the 5 contexts from Figshare (Prat et al. 2017, https://doi.org/10.1038/sdata.2017.143, CC BY 4.0) into `data/audio/` (50,998 recordings, about 54 GB).
- `prepare.py`: first 2 seconds of each annotated call become a 128x256 spectrogram (0-62.5 kHz). The last 10% of recording days are the test set (fold -1, 6,569 calls). The other days are divided into 5 folds (0-4, about 9,240 calls each) with `StratifiedGroupKFold`: a day is never split, and every fold has a similar mix of contexts. Writes `data/x.npy` and `data/y.csv`.
- `train.py`: 5-fold cross-validation. For each fold a fresh ResNet18 (pretrained on ImageNet, all layers trained) is trained on the other 4 folds and checked on the held-out fold. Prints train and validation accuracy per epoch and one per-context report over all 5 held-out folds (46,200 calls).

The folds mix days from all months, so a held-out day has neighbouring days in training. The cross-validation score is therefore higher than a score on later days (the test set) will be. It is for comparing fixes, not a prediction of the test score.

Run with `.venv\Scripts\python <script>`. `data/` is not in git.

## Log: problems and chosen fixes

| # | Problem | Fix chosen | Validation result |
|---|---|---|---|
| 0 | First version, nothing fixed yet (plain training, 5 epochs, one validation block = days 80-90%) | none | accuracy 0.758, macro-F1 0.440 after epoch 5; best epoch was 3 (accuracy 0.790); train accuracy reached 0.938. Always answering "Sleeping" scores 0.699. Recall: Biting 0.05, Feeding 0.70, Fighting 0.37, Mating protest 0.12, Sleeping 0.90 |
| 1 | The student asked for cross-validation instead of one validation block | 5-fold cross-validation by recording day | accuracy 0.801, macro-F1 0.580 after epoch 5 (always "Sleeping": 0.640). Mean validation accuracy per epoch: 0.804, 0.808, 0.812, 0.802, 0.802; train accuracy reaches about 0.95. Recall: Biting 0.04, Feeding 0.76, Fighting 0.64, Mating protest 0.44, Sleeping 0.91 |
| 2 | Too few non-Sleeping calls (student's idea: add a copy of every non-Sleeping training call rotated by 90 degrees, squeezed back to 128x256) | tested with a separate copy of `train.py`; not in the code (the student has not decided yet) | accuracy 0.785, macro-F1 0.563 after epoch 5; mean validation accuracy per epoch 0.799, 0.812, 0.811, 0.806, 0.785. Recall: Biting 0.02, Feeding 0.85, Fighting 0.61, Mating protest 0.54, Sleeping 0.87. No gain over row 1 |

Still open: the model memorizes (train accuracy keeps rising while validation accuracy peaks at epoch 3 and the reported model is from epoch 5), and Biting is almost never recognized.

# Bat calls

Deep Learning class project (team of 3). A model listens to a call of an Egyptian fruit bat and predicts what the bats were doing: Biting, Feeding, Fighting, Mating protest or Sleeping.

The student driving this repo is new to deep learning and must be able to explain every line in class.

## How to work here

- Keep everything as simple as possible: three short scripts, no comments or docstrings in the code, no extra files or folders.
- One problem at a time. After a run, say in plain language what the problem is, offer 2-3 fixes with their trade-offs and a recommendation, then stop. **The student chooses the fix.** Don't apply fixes that weren't chosen, and don't combine several changes in one step.
- Explain in the chat, not in the code. Never fake results; say when a difference is too small to mean anything.
- Compare fixes on the validation days only. The test days are used once, at the very end.
- Commit each chosen fix separately and push, then add a row to the log below.

## The pieces

- `download.py`: gets the labels and every recording of the 5 contexts from Figshare (Prat et al. 2017, https://doi.org/10.1038/sdata.2017.143, CC BY 4.0) into `data/audio/` (50,998 recordings, about 54 GB).
- `prepare.py`: first 2 seconds of each annotated call become a 128x256 spectrogram (0-62.5 kHz). Split by recording day: first 80% of days train, next 10% validation, last 10% test. Writes `data/x.npy` and `data/y.csv`.
- `train.py`: ResNet18 pretrained on ImageNet, all layers trained. Prints train and validation accuracy per epoch and a per-context report on the validation days.

Run with `.venv\Scripts\python <script>`. `data/` is not in git.

## Log: problems and chosen fixes

| # | Problem | Fix chosen | Validation result |
|---|---|---|---|
| 0 | First version, nothing fixed yet (plain training, 5 epochs) | none | accuracy 0.758, macro-F1 0.440 after epoch 5; best epoch was 3 (accuracy 0.790); train accuracy reached 0.938. Always answering "Sleeping" scores 0.699. Recall: Biting 0.05, Feeding 0.70, Fighting 0.37, Mating protest 0.12, Sleeping 0.90 |

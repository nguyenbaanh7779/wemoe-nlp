# Setup Guide

The experiments need a GPU and run on Google Colab through [`notebooks/colab_run.ipynb`](../../notebooks/colab_run.ipynb). A local environment is only needed for the unit tests and for reading results.

## Local Environment

The local Mac setup has no GPU, so it is only used for the CPU unit tests (tiny random BERTs, offline, a few seconds) and for summarising results downloaded from Colab.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

`requirements.txt` includes `torch`. On Colab it is already installed, so that line is a no-op there.

## Running on Colab

### 1. Push the code

The notebook clones the repository from GitHub, so it only sees commits that have been pushed:

```bash
git push -u origin <branch>
```

The clone cell uses `BRANCH = 'main'`. To run a branch that has not been merged yet, set `BRANCH` to that branch name in the clone cell.

If the repository is private, add a Colab secret (🔑 in the left sidebar) named `GITHUB_TOKEN`. It should be a fine-grained, read-only token for this repository, with "Notebook access" enabled. Never paste a token into a cell.

### 2. Open the notebook

Use either way:

- **File → Upload notebook**, then pick `notebooks/colab_run.ipynb` from your machine. Colab saves the uploaded copy to `MyDrive/Colab Notebooks/colab_run.ipynb`, and all later edits and outputs go to that copy. The local file is not changed. Reopen the Drive copy next time; uploading again creates a new copy without the old outputs.
- **File → Open notebook → GitHub**, then pick the repository and branch. This copy is **not saved anywhere** until you choose **File → Save a copy in Drive**. If you close the tab before saving, you lose your edits and all outputs.

Then select **Runtime → Change runtime type → T4 GPU** (or better).

### 3. Run the cells in order

| Cell | What it does | Time on a T4 (rough) |
|---|---|---|
| `nvidia-smi` | checks that a GPU is attached | — |
| Mount Drive | sets `DRIVE_ROOT = /content/drive/MyDrive/wemoe-nlp` | — |
| Clone + install | clones the repository into `/content/wemoe-nlp`, then runs `pip install -r requirements.txt` | 1–2 min |
| `pytest` | all tests must pass | < 1 min |
| Smoke run (optional) | runs the whole pipeline on tiny subsets ([`configs/smoke.yaml`](../../configs/smoke.yaml)); the numbers are meaningless | a few min |
| Experiment 1 | trains the five single-task experts | about 45 min (QNLI alone about 30 min) |
| Baselines | Simple Averaging, and Task Arithmetic with λ chosen on dev | about 2 min |
| Experiment 2 | WEMoE + FSFT, seeds 0, 1, 2 | about 16 min per run, 50 min in total |
| k sweep + scope ablation | k ∈ {16, 64}, and `moe_scope=intermediate` | about 2 h (intermediate runs take about 8.5 min each) |
| Router ablation | `router_hidden_layers=0`: input-independent, layer-wise coefficients | about 50 min |
| Experiment 4 | parameter counts, storage, latency | a few min |
| Summary | writes `results/summary.md` | — |

These times are from the first full run on a T4 (`notebooks/colab_run_executed_v1.ipynb`). The FSFT time does not depend on k, because every run trains for 500 steps.

The steps depend on each other:
- Baselines need the trained experts.
- WEMoE starts its routers from the λ stored in `results/baselines.json`.

Every step also skips work that is already done:
- an expert counts as done once its checkpoint and `results/experts/<task>.json` exist;
- a WEMoE run counts as done once its `results.json` exists.

So after a disconnect or a lost runtime, reconnect and rerun the notebook from the top. To redo a step, pass `--overwrite`.

### Useful command-line options

Every script reads [`configs/default.yaml`](../../configs/default.yaml). Any value can be overridden with `--set section.key=value`:

```bash
!python scripts/train_expert.py {CFG} --tasks mrpc rte                    # only some tasks
!python scripts/train_wemoe.py {CFG} --set wemoe.k=64 --set wemoe.seed=1  # change any config value
!python scripts/train_wemoe.py {CFG} --set wemoe.impl=decomposed          # uses less GPU memory
!python scripts/train_wemoe.py {CFG} --overwrite                          # rerun a finished run
```

## Outputs

The notebook writes everything to Google Drive:

```
MyDrive/wemoe-nlp/
├── checkpoints/
│   ├── experts/<task>/   # fine-tuned BERT experts, about 440 MB each (about 2.2 GB in total)
│   └── wemoe/<run>/      # router.pt (router weights only) and wemoe_config.json
└── results/              # json, jsonl and summary.md; this is the part to download
```

The smoke run writes to `/content/smoke` on the Colab VM, so it never mixes with the real results. The free Drive plan has 15 GB, which is enough, but check the free space before Experiment 1.

See the [Outputs table in the main README](../../README.md#outputs) for what each results file contains.

## Getting Results Back

### Results

You normally don't need the checkpoints locally. Download only `results/`. Then unzip it into the repository root, so the files end up in `wemoe-nlp/results/`.

- **Browser:** on drive.google.com, right-click `wemoe-nlp/results` → **Download**. Drive downloads it as a `.zip`.
- **From Colab:** add a cell at the end of the notebook:

  ```python
  !cd /content/drive/MyDrive/wemoe-nlp && zip -qr /content/results.zip results
  from google.colab import files
  files.download('/content/results.zip')
  ```

  Then, on your machine:

  ```bash
  unzip ~/Downloads/results.zip -d ~/Documents/projects/wemoe-nlp
  ```

- **Google Drive for desktop:** Drive appears in Finder, and you copy `results/` from there.

To rebuild the tables and the report figures locally:

```bash
python scripts/summarize_results.py --results-dir results   # results/summary.md
python scripts/plot_results.py --results-dir results        # docs/report/figures/*.png
```

The routing heatmap needs `results/wemoe/full-mean-k256-s0/predictions_validation.jsonl`, so download `results/` with the prediction files. The results are written up in the [report](../report/README.md).

### Notebook

**File → Download → Download .ipynb**. Save it as `notebooks/colab_run_executed.ipynb` to keep the outputs and leave the clean template unchanged. Download it after every long run: the Drive copy keeps the outputs, but the GitHub-opened copy does not.

### Committing

`.gitignore` already excludes the checkpoints (`checkpoints/`, `*.safetensors`, `*.pt`) and the large prediction files (`results/**/*.jsonl`). `git add results/ notebooks/` therefore adds only the small files: `summary.md`, the `.json` results and the notebook.

## Troubleshooting

- **`Could not open requirements file: requirements.txt` and `git log` shows `Initial commit`.** `/content/wemoe-nlp` already exists from an earlier clone of a different branch. The clone cell only clones when that folder is missing, so changing `BRANCH` has no effect. Delete the old clone and rerun the clone cell:

  ```python
  %cd /content
  !rm -rf /content/wemoe-nlp
  ```

  This deletes only the copy on the VM, not your Drive data.
- **`git clone failed`.** The branch has not been pushed, the branch name is misspelled, or the repository is private and the `GITHUB_TOKEN` secret is missing or has no notebook access.
- **CUDA out of memory during WEMoE training.** Add `--set wemoe.impl=decomposed`. If that is not enough, also lower the batch size, for example `--set wemoe.batch_size=8`.
- **Runtime disconnected in the middle of a step.** Rerun the notebook from the top. Finished experts and runs are skipped. An unfinished run starts again from the beginning.
- **`Normalized` and `Gain over majority` show `-` in the summary.** `results/baselines.json` is missing, so there are no single-task scores to compare with. Run the baselines step first. Older `baselines.json` files have no majority-class scores; the summary then reads the labels from any `predictions_validation.jsonl`, and rerunning the baselines step (about 2 min) stores them.

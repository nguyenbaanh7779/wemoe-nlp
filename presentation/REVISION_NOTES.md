# Revision notes: `presentation_revised`

Originals are unchanged (`presentation.tex`, `presentation.pdf`, `references.bib`, `report-ref/*`). `script_vi.md` was updated in place on request.

## Files

| File | Purpose |
|---|---|
| `presentation_revised.tex` | Editable source, HUST theme with the same options as the original |
| `presentation_revised.pdf` | Compiled deck, 24 pages |
| `presentation_revised.pptx` | PowerPoint export: each slide is one page image, so it is not editable |
| `references_revised.bib` | Six references (original had one) |
| `figures/{wemoe,bert,transformer}_paper.png` | Crops of `report-ref/*.jpg` without the Vietnamese captions |

Build from `presentation/`:

```bash
export TEXINPUTS=../../HUST-Beamer-Theme//:
lualatex presentation_revised && biber presentation_revised && lualatex presentation_revised && lualatex presentation_revised
```

## Structure

| Part | Original | Revised | Change |
|---|---|---|---|
| Main slides | 13 | 15 | All titles and order kept; slides 5 and 6 added (+15%) |
| Backup slides | 2 | 8 | Both kept and shortened; 6 added after the conclusion |
| References | 1 | 1 | Kept |

## Audit: original deck against thesis and code

| Topic | Thesis (`report-ref/*.pdf`) | Code | Original deck | Action |
|---|---|---|---|---|
| Router input | Mean of `r(h)`, Eq. 2.2 | Router per token, then masked mean (`wemoe.py:195-203`) | "uses the mean token representation", "mean-pooled inputs" | **Corrected** |
| Router depth | Two layers, Eq. 2.3 | `fc2(relu(fc1(h)))` (`wemoe.py:68-97`) | "two hidden layers" | **Corrected**: two linear layers with ReLU |
| Router output | No softmax in Eq. 2.2-2.3 | Unnormalised, bias at lambda | Unnormalised | Kept |
| Upscaled scope | `intermediate` only | `full` by default; `intermediate` as ablation | Both | Kept; difference on backup slide |
| Adapter-WEMoE | Proposed | Not implemented | Absent | Excluded from main story; noted on backup slide |
| Router training | TTA and FSFT | FSFT only (`fsft.py`) | FSFT | Kept; loss now written out |
| Experts at inference | Not explicit | Not run: only task vectors and heads stored | Not explicit | Stated on slide 4 |
| Results | Own tables, per-task hyperparameters | `results/summary.md`, `docs/report/README.md` | From repository | Unchanged; thesis numbers not used |

## Request items adapted to the implementation

| Request wording | Implementation | On the slides |
|---|---|---|
| Router outputs a probability distribution | Real-valued weights, may be negative | "No softmax, no top-k" |
| Weights combine full experts to a final prediction | Task vectors are mixed inside MLP layers, per layer | Single merged forward pass |
| Regression heads | All five tasks are binary classification | Stated on the fine-tuning backup |
| Sparse/dense routing | Dense: all five task vectors in every layer | Router backup; top-k only in classic MoE |
| Logit ensembling | Not implemented, no measurements | Labelled as concept; qualitative comparison only |

## Speaker script

`script_vi.md` now follows `presentation_revised.pdf` (15 main slides, 8 backups) and no longer matches the numbering of the original `presentation.pdf`.

## Verification against the current repository

Run commit `35f8552` vs. `HEAD`: `wemoe.py`, `merging.py`, `data.py`, `experts.py` and `configs/` are unchanged; `fsft.py` and `baselines.py` differ only in run naming and majority reporting. `pytest`: 44 passed.

| Evidence level | Claims |
|---|---|
| Checked against code or computed | Router, merge, loss and optimiser settings; stored and trainable parameter counts; `bert-base-uncased` config; averages and gains from the per-task means |
| Checked against local data or results | Majority accuracy and validation sizes (cached GLUE); baselines; `results.json` of the 7 local runs |
| Checked against the executed notebook | k=64 seed 2 and intermediate seed 2 scores and best steps |
| Report only, cannot be recomputed locally | Full-MLP k=256 mean (seeds 1, 2 missing; local seed 0 is 69.7), per-task std values, routing statistics (`predictions_validation.jsonl` missing) |
| External background, not repository behaviour | Backup slides on Transformer, BERT pre-training and classic MoE; "MLPs change most after fine-tuning" on the sources backup |

## Number policy

Every result or model figure on the slides and in `script_vi.md` must come from this repository (`results/`, `docs/report/README.md`, `configs/`, `src/`). Removed under this rule: 768, 3072, the 15% masking rate, the positional-encoding constant, and thesis equation numbers. Bibliography years and slide numbers are not data.

## Wording policy

Slides and `script_vi.md` must not mention the repository or the original thesis/report. The sources backup is now "WEMoE paper and this project", and the thesis is no longer in `references_revised.bib`.

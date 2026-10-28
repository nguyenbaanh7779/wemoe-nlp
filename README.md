# wemoe-nlp

Reproduction and analysis of **Weight-Ensembling Mixture of Experts** (WEMoE; Tang et al., ICML 2024) for multi-task NLP.
Five BERT-base experts (MRPC, RTE, CoLA, SST-2, QNLI) are merged into one model. In that model, every MLP layer is an
input-dependent mixture of the experts' task vectors. The routers are trained with **Few-Shot Fine-Tuning (FSFT)**.

## Differences from the paper

| | WEMoE paper | This project |
|---|---|---|
| Backbone and tasks | CLIP-ViT, 8 image classification tasks | BERT-base, 5 GLUE tasks |
| Router training | test-time adaptation (entropy minimisation on unlabelled test data) | FSFT: cross-entropy on *k* labelled examples per task, held out from expert training |
| Upscaled modules | Transformer MLPs | MLPs (`intermediate.dense` + `output.dense`), one router per layer; intermediate-only as an ablation |
| Routing input | mean of token-level routing weights | same, over non-padding tokens only |
| Routing weights | unnormalised (no softmax), bias initialised to λ | same |
| Other parameters | Task Arithmetic, λ = 0.3 | Task Arithmetic, λ chosen on dev |

The paper has no GLUE results, so it is only compared qualitatively.

## Data protocol

- Each task's official GLUE **train** split is divided by a fixed seed into three parts:
  - `fewshot_pool`: 256 examples;
  - `dev`: 10%, at most 1000 examples;
  - `expert_train`: the rest.
- The experts never see `dev` or the few-shot pool.
- All model selection uses `dev`: the expert epoch, the Task Arithmetic λ and early stopping of the routers.
- The official GLUE **validation** split is only used for the reported numbers, for every task including MRPC.

## Layout

```
configs/         default.yaml (all experiments, one shared data split), smoke.yaml (tiny end-to-end check)
src/wemoe_nlp/   data, metrics, merging, wemoe, experts, baselines, fsft, efficiency, inference
scripts/         train_expert, eval_baselines, train_wemoe, count_params, summarize_results
notebooks/       colab_run.ipynb: the whole pipeline on a Colab GPU, results saved to Google Drive
tests/           CPU unit tests on tiny random BERTs (offline, a few seconds)
```

## Running

On Colab, open `notebooks/colab_run.ipynb`.

Locally (a GPU is needed for the real runs):

```bash
pip install -r requirements.txt
python -m pytest -q
python scripts/train_expert.py                       # Experiment 1: single-task experts
python scripts/eval_baselines.py                     # Simple Averaging, Task Arithmetic (λ on dev)
python scripts/train_wemoe.py --set wemoe.seed=0     # Experiment 2: WEMoE + FSFT (repeat for seeds 1, 2)
python scripts/count_params.py --latency             # Experiment 4: parameters, storage, latency
python scripts/summarize_results.py                  # tables in results/summary.md
```

Use `--set section.key=value` to override any config value, for example
`--set wemoe.k=64 --set wemoe.moe_scope=intermediate`. Use `--config configs/smoke.yaml` for a quick pipeline check.

## Outputs

| Path | Content |
|---|---|
| `checkpoints/experts/<task>/` | fine-tuned expert (`save_pretrained`) |
| `checkpoints/wemoe/<run>/` | `router.pt` (router weights only) and `wemoe_config.json` |
| `results/experts/<task>.json` | split sizes, dev history, validation metrics |
| `results/baselines.json` | single-task, Simple Averaging, Task Arithmetic λ grid and selected λ |
| `results/wemoe/<run>/results.json` | config, best step, validation metrics at init and after FSFT, parameter counts |
| `results/wemoe/<run>/history.jsonl` | loss, gradient norm and dev metrics per step |
| `results/wemoe/<run>/predictions_validation.jsonl` | per example: text, label, prediction, probabilities, routing weights `[layer][expert]` |
| `results/efficiency.json`, `results/summary.md` | Experiment 4 and the result tables |

The prediction files feed the routing analysis (Experiment 5) and the error analysis (Experiment 6).
`fsft.load_trained_wemoe` rebuilds a trained model from the experts and `router.pt`.

## Expected size (BERT-base, five tasks)

| Deployment | Stored parameters | vs. five models |
|---|---:|---:|
| 5 single-task models | 547.4M | 100% |
| WEMoE, full MLP | 400.0M | 73.1% |
| WEMoE, intermediate only | 258.4M | 47.2% |
| Task Arithmetic | 109.5M | 20.0% |

WEMoE keeps the pre-trained MLPs plus one task vector per expert for every upscaled layer, so it removes the
duplicated non-MLP weights but not all of the per-task storage. Each input also needs its own merged MLP weights.
`scripts/count_params.py` measures these numbers on the built models.

## Credits

- WEMoE: Anke Tang et al., *Merging Multi-Task Models via Weight-Ensembling Mixture of Experts*, ICML 2024
  ([arXiv:2402.00433](https://arxiv.org/abs/2402.00433)). The router and weight-ensembling module are adapted from
  [tanganke/weight-ensembling_MoE](https://github.com/tanganke/weight-ensembling_MoE).
- Earlier implementation: HUST_MoE / DANC code by Thang Tran Ngoc, Hong Son Nguyen and Ba Anh Nguyen.

# Results (official GLUE validation splits)

| Model | MRPC (acc/F1) | RTE (acc) | CoLA (MCC) | SST-2 (acc) | QNLI (acc) | Avg score | Normalized | Gain over majority |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Single-task experts | 82.1/88.1 | 63.2 | 59.1 | 92.2 | 91.2 | 78.2 | 100.0 | 100.0 |
| Majority class | 68.4/81.2 | 52.7 | 0.0 | 50.9 | 50.5 | 45.8 | 56.4 | 0.0 |
| Simple Averaging (λ=1/5) | 68.4/81.2 | 46.9 | 23.3 | 55.4 | 65.9 | 53.3 | 66.8 | 6.6 |
| Task Arithmetic (λ=0.3) | 68.6/81.3 | 47.3 | 16.9 | 69.8 | 77.9 | 57.4 | 70.6 | 18.4 |
| WEMoE init (full, mean, h=0, λ=0.3, seeds=[0, 1, 2]) | 68.6±0.0/81.3±0.0 | 47.3±0.0 | 16.9±0.0 | 69.8±0.0 | 77.9±0.0 | 57.4±0.0 | 70.6±0.0 | 18.4±0.0 |
| WEMoE + FSFT (full, mean, h=0, k=256, lr=0.001, seeds=[0, 1, 2]) | 68.4±0.0/81.2±0.0 | 47.3±0.0 | 20.2±0.0 | 70.9±0.1 | 78.6±0.0 | 58.4±0.0 | 72.0±0.0 | 20.0±0.0 |
| WEMoE init (full, mean, λ=0.3, seeds=[0, 1, 2]) | 68.5±0.1/81.3±0.1 | 47.1±0.2 | 19.3±2.0 | 70.2±1.1 | 77.7±0.2 | 57.8±0.6 | 71.3±1.0 | 18.7±1.5 |
| WEMoE + FSFT (full, mean, k=16, lr=0.0001, seeds=[0, 1, 2]) | 68.7±2.3/80.6±2.2 | 48.0±1.0 | 27.6±14.1 | 79.6±8.1 | 74.0±6.5 | 60.8±2.7 | 75.6±4.6 | 25.5±4.6 |
| WEMoE + FSFT (full, mean, k=64, lr=0.0001, seeds=[0, 1, 2]) | 71.6±4.4/80.9±4.3 | 54.0±3.4 | 20.5±17.8 | 89.0±0.6 | 83.1±2.4 | 64.6±3.7 | 79.5±6.3 | 46.7±8.5 |
| WEMoE + FSFT (full, mean, k=256, lr=0.0001, seeds=[0, 1, 2]) | 76.3±4.5/83.9±2.8 | 52.0±4.4 | 42.6±0.9 | 88.0±0.5 | 86.7±0.7 | 69.9±0.1 | 87.8±0.5 | 59.1±1.1 |
| WEMoE init (intermediate, mean, λ=0.3, seeds=[0, 1, 2]) | 68.5±0.1/81.3±0.1 | 47.1±0.2 | 18.1±1.4 | 69.9±0.5 | 77.8±0.2 | 57.5±0.4 | 70.8±0.6 | 18.1±1.2 |
| WEMoE + FSFT (intermediate, mean, k=256, lr=0.0001, seeds=[0, 1, 2]) | 66.5±0.7/76.6±1.0 | 59.0±0.9 | 41.7±0.9 | 86.7±0.2 | 86.5±0.2 | 69.1±0.1 | 87.4±0.1 | 54.8±1.7 |

Scores ×100; mean±std over seeds. Avg score: mean of task scores (MRPC: mean of acc and F1, CoLA: MCC). Normalized: task score / single-task score, averaged over tasks. Gain over majority: (score − majority) / (single-task − majority), averaged over tasks; 0 is no better than always predicting the most frequent label.

## Training cost

| Run | Best step / epoch | Train time (min) |
|---|---:|---:|
| Expert mrpc | epoch 2 | 0.8 |
| Expert rte | epoch 3 | 0.7 |
| Expert cola | epoch 3 | 1.1 |
| Expert sst2 | epoch 2 | 11.0 |
| Expert qnli | epoch 3 | 29.8 |
| WEMoE (full, mean, h=0, k=256) | 50, 50, 50 of 500 | 15.4 |
| WEMoE (full, mean, k=16) | 50, 0, 150 of 500 | 16.3 |
| WEMoE (full, mean, k=64) | 100, 250, 100 of 500 | 14.4 |
| WEMoE (full, mean, k=256) | 250, 200, 200 of 500 | 16.3 |
| WEMoE (intermediate, mean, k=256) | 500, 400, 400 of 500 | 8.4 |

Train time includes the dev evaluations; WEMoE times are means over seeds.

## Parameters, storage and latency

| Deployment | # checkpoints | Stored params | Size (fp32) | vs 5 models |
|---|---:|---:|---:|---:|
| 5 single-task models | 5 | 547.4M | 2.04 GiB | 100.0% |
| Task Arithmetic | 1 | 109.5M | 0.41 GiB | 20.0% |
| WEMoE (full MLP) | 1 | 400.0M | 1.49 GiB | 73.1% |
| WEMoE (intermediate only) | 1 | 258.4M | 0.96 GiB | 47.2% |

| Deployment | batch 1: ms/example (peak MB) | batch 32: ms/example (peak MB) |
|---|---:|---:|
| 5 single-task models | 11.64 (614) | 1.61 (704) |
| Task Arithmetic | 12.67 (435) | 1.63 (519) |
| WEMoE (full MLP) | 27.83 (1615) | 7.31 (1987) |
| WEMoE (intermediate only) | 18.90 (1061) | 4.69 (1387) |

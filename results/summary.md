# Results (official GLUE validation splits)

| Model | MRPC (acc/F1) | RTE (acc) | CoLA (MCC) | SST-2 (acc) | QNLI (acc) | Avg score | Normalized |
|---|---:|---:|---:|---:|---:|---:|---:|
| Single-task experts | 82.1/88.1 | 63.2 | 59.1 | 92.2 | 91.2 | 78.2 | 100.0 |
| Simple Averaging (λ=1/5) | 68.4/81.2 | 46.9 | 23.3 | 55.4 | 65.9 | 53.3 | 66.8 |
| Task Arithmetic (λ=0.3) | 68.6/81.3 | 47.3 | 16.9 | 69.8 | 77.9 | 57.4 | 70.6 |
| WEMoE init (full, λ=0.3) | 68.5±0.1/81.3±0.1 | 47.1±0.2 | 19.3±2.0 | 70.2±1.1 | 77.7±0.2 | 57.8±0.6 | 71.3±1.0 |
| WEMoE + FSFT (full, mean, k=16, seeds=[0, 1, 2]) | 68.7±2.3/80.6±2.2 | 48.0±1.0 | 27.6±14.1 | 79.6±8.1 | 74.0±6.5 | 60.8±2.7 | 75.6±4.6 |
| WEMoE init (full, λ=0.3) | 68.5±0.1/81.3±0.1 | 47.1±0.2 | 19.3±2.0 | 70.2±1.1 | 77.7±0.2 | 57.8±0.6 | 71.3±1.0 |
| WEMoE + FSFT (full, mean, k=64, seeds=[0, 1, 2]) | 71.6±4.4/80.9±4.3 | 54.0±3.4 | 20.5±17.8 | 89.0±0.6 | 83.1±2.4 | 64.6±3.7 | 79.5±6.3 |
| WEMoE init (full, λ=0.3) | 68.5±0.1/81.3±0.1 | 47.1±0.2 | 19.3±2.0 | 70.2±1.1 | 77.7±0.2 | 57.8±0.6 | 71.3±1.0 |
| WEMoE + FSFT (full, mean, k=256, seeds=[0, 1, 2]) | 76.3±4.5/83.9±2.8 | 52.0±4.4 | 42.6±0.9 | 88.0±0.5 | 86.7±0.7 | 69.9±0.1 | 87.8±0.5 |
| WEMoE init (intermediate, λ=0.3) | 68.5±0.1/81.3±0.1 | 47.1±0.2 | 18.1±1.4 | 69.9±0.5 | 77.8±0.2 | 57.5±0.4 | 70.8±0.6 |
| WEMoE + FSFT (intermediate, mean, k=256, seeds=[0, 1, 2]) | 66.5±0.7/76.6±1.0 | 59.0±0.9 | 41.7±0.9 | 86.7±0.2 | 86.5±0.2 | 69.1±0.1 | 87.4±0.1 |

Scores ×100; mean±std over seeds. Avg score: mean of task scores (MRPC: mean of acc and F1, CoLA: MCC). Normalized: task score / single-task score, averaged over tasks.

## Parameters and storage

| Deployment | # checkpoints | Stored params | vs 5 models |
|---|---:|---:|---:|
| single_task_models | 5 | 547.4M | 100.0% |
| task_arithmetic | 1 | 109.5M | 20.0% |
| wemoe_full | 1 | 400.0M | 73.1% |
| wemoe_intermediate | 1 | 258.4M | 47.2% |

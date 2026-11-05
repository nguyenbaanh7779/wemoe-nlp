# TODO

Results and context for these items are in the [report](docs/report/README.md).

## Planned experiments (new branch)

- [ ] **Merge four tasks without RTE.** This tests whether RTE interferes with the other tasks. The routers subtract the RTE task vector in layers 11–12 for every input (report §6).
  - Reuse the four existing experts with `--tasks mrpc cola sst2 qnli`.
  - Rerun the baselines so that λ is chosen for four tasks.
  - Run FSFT with k = 256 and seeds 0–2. Everything takes about 1 h on a T4.
  - Use a separate `paths.results_dir` and `wemoe.run_name`. Otherwise the runs are skipped because the names match, or they overwrite the five-task routers in `checkpoints/wemoe/`.
  - Report it as a supplementary result next to the five-task results, not in their place.
- [ ] **Larger few-shot budgets: k = 256 → 512 → 1024.** The k sweep has not levelled off (report §4), and even at k = 256 the routers overfit after step 200–250.
  - Split the data once with `fewshot_pool_size=1024`.
  - Cap the RTE pool at about 512, since RTE has only 2490 training pairs. This needs a per-task pool size in the code.
  - Retrain the experts once (about 45 min).
  - Run k ∈ {256, 512, 1024}, raising `max_steps` to 1000–1500 for the larger k. This takes about 4–5 h.
  - Fallback: a pool of 512 for every task, with k ∈ {256, 512}.
  - Ruled out: using the GLUE validation split as extra training data, for RTE or any other task. It is the only clean reporting split.
- [ ] **Task-vector norms per module type** (attention, `intermediate.dense`, `output.dense`). This checks the reason for upscaling only `intermediate.dense`, as in DANC's L2 plot.
  - Keep `attention.output.dense` apart from the MLP `output.dense` when filtering by name.

## Other follow-ups

- [ ] Compare the per-task results with the earlier DANC project in more depth, if the report needs it (report §11).
- [ ] A strict few-shot variant with dev capped at k.
- [ ] More FSFT steps for the intermediate-only model.
- [ ] Analyse routing per example: does within-task variation follow sentence length, label or errors? Use `predictions_validation.jsonl`.
- [ ] Build the demo.

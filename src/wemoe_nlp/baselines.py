"""Static baselines on the official validation split: single-task experts, Simple Averaging and
Task Arithmetic (lambda chosen on the dev split, never on validation)."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import torch

from .data import Collator, DataConfig, load_task, make_loader
from .experts import check_expert_data_config, load_experts, load_pretrained, load_tokenizer
from .inference import evaluate_tasks
from .merging import TaskArithmeticMerger, expert_as_multihead
from .metrics import gain_over_majority, majority_metrics, mean_score, normalized_scores
from .utils import library_versions, save_json

logger = logging.getLogger(__name__)


@dataclass
class BaselineConfig:
    lambdas: list = field(default_factory=lambda: [round(0.1 * i, 1) for i in range(1, 11)])
    eval_batch_size: int = 64
    fp16: bool = True


def baselines_path(results_dir) -> Path:
    return Path(results_dir) / "baselines.json"


def run_baselines(
    tasks: Sequence[str],
    data_cfg: DataConfig,
    cfg: BaselineConfig,
    checkpoint_dir,
    results_dir,
    device: torch.device,
) -> dict:
    tasks = list(tasks)
    check_expert_data_config(results_dir, tasks, data_cfg)
    tokenizer = load_tokenizer(data_cfg.model_name)
    collator = Collator(tokenizer.pad_token_id)
    data = {t: load_task(t, tokenizer, data_cfg, splits=("dev", "validation")) for t in tasks}
    dev_loaders = {t: make_loader(d["dev"], collator, cfg.eval_batch_size) for t, d in data.items()}
    val_loaders = {t: make_loader(d["validation"], collator, cfg.eval_batch_size) for t, d in data.items()}
    amp = cfg.fp16 and device.type == "cuda"

    pretrained = load_pretrained(data_cfg.model_name)
    experts = load_experts(checkpoint_dir, tasks)

    single_dev, single_val = {}, {}
    for task in tasks:
        model = expert_as_multihead(task, experts[task]).to(device)
        single_dev[task] = evaluate_tasks(model, {task: dev_loaders[task]}, device, amp=amp)[task]
        single_val[task] = evaluate_tasks(model, {task: val_loaders[task]}, device, amp=amp)[task]
        model.cpu()
        logger.info("single-task %s: validation %s", task, single_val[task])

    merger = TaskArithmeticMerger(pretrained, experts).to(device)
    grid = []
    for lam in cfg.lambdas:
        dev = evaluate_tasks(merger.apply(lam), dev_loaders, device, amp=amp)
        grid.append({"lambda": lam, "dev": dev, "dev_mean_score": mean_score(dev)})
        logger.info("task arithmetic lambda=%.2f: dev mean score %.4f", lam, grid[-1]["dev_mean_score"])
    best = max(grid, key=lambda row: row["dev_mean_score"])
    ta_val = evaluate_tasks(merger.apply(best["lambda"]), val_loaders, device, amp=amp)

    sa_lambda = 1.0 / len(tasks)
    sa_model = merger.apply(sa_lambda)
    sa_dev = evaluate_tasks(sa_model, dev_loaders, device, amp=amp)
    sa_val = evaluate_tasks(sa_model, val_loaders, device, amp=amp)
    majority_val = {t: majority_metrics(t, d["validation"]["labels"]) for t, d in data.items()}

    results = {
        "tasks": tasks,
        "data_config": data_cfg.to_dict(),
        "config": asdict(cfg),
        "single_task": {
            "dev": single_dev,
            "validation": single_val,
            "validation_mean_score": mean_score(single_val),
        },
        "majority": {
            "validation": majority_val,
            "validation_mean_score": mean_score(majority_val),
            "normalized": normalized_scores(majority_val, single_val),
        },
        "simple_average": {
            "lambda": sa_lambda,
            "dev": sa_dev,
            "validation": sa_val,
            "validation_mean_score": mean_score(sa_val),
            "normalized": normalized_scores(sa_val, single_val),
            "gain_over_majority": gain_over_majority(sa_val, single_val, majority_val),
        },
        "task_arithmetic": {
            "lambda_grid": grid,
            "best_lambda": best["lambda"],
            "validation": ta_val,
            "validation_mean_score": mean_score(ta_val),
            "normalized": normalized_scores(ta_val, single_val),
            "gain_over_majority": gain_over_majority(ta_val, single_val, majority_val),
        },
        "versions": library_versions(),
    }
    save_json(results, baselines_path(results_dir))
    logger.info("best task arithmetic lambda %.2f; results in %s", best["lambda"], baselines_path(results_dir))
    return results

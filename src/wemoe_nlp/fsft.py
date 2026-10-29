"""Few-Shot Fine-Tuning (FSFT) of the WEMoE routers.

Only the routers are trained, with cross-entropy on ``k`` labelled examples per task drawn from the
few-shot pool (examples the experts never saw). This is a supervised variant of WEMoE: the paper
trains the routers by test-time adaptation (entropy minimisation on unlabelled test data).
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .baselines import baselines_path
from .data import TASKS, Collator, DataConfig, load_task, make_loader, model_inputs, sample_fewshot_indices, task_seed
from .experts import check_expert_data_config, load_experts, load_pretrained, load_tokenizer
from .inference import evaluate_tasks, predict
from .metrics import compute_metrics, mean_score, normalized_scores
from .utils import (autocast, library_versions, load_json, make_grad_scaler, save_json, set_seed,
                    stored_numel, trainable_numel, write_jsonl)
from .wemoe import WEMoEConfig, build_wemoe, parameter_breakdown

logger = logging.getLogger(__name__)


@dataclass
class FSFTConfig:
    k: int = 256  # labelled examples per task
    seed: int = 0
    init_lambda: Optional[float] = None  # None: best Task Arithmetic lambda from baselines.json
    moe_scope: str = "full"
    routing: str = "mean"
    router_hidden_layers: int = 2
    impl: str = "merged"
    lr: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 16
    eval_batch_size: int = 64
    max_steps: int = 500
    warmup_ratio: float = 0.1
    eval_every: int = 50
    max_grad_norm: Optional[float] = None
    backbone_dropout: bool = False  # dropout inside the frozen BERT while training the routers
    fp16: bool = True
    run_name: Optional[str] = None

    def resolve_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        layers = "" if self.router_hidden_layers == 2 else f"-h{self.router_hidden_layers}"
        return f"{self.moe_scope}-{self.routing}{layers}-k{self.k}-s{self.seed}"


def run_dirs(checkpoint_dir, results_dir, run_name: str):
    return Path(checkpoint_dir) / "wemoe" / run_name, Path(results_dir) / "wemoe" / run_name


def resolve_init_lambda(cfg: FSFTConfig, results_dir) -> float:
    if cfg.init_lambda is not None:
        return float(cfg.init_lambda)
    path = baselines_path(results_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"wemoe.init_lambda is null and {path} does not exist: run scripts/eval_baselines.py first "
            "or set wemoe.init_lambda explicitly"
        )
    return float(load_json(path)["task_arithmetic"]["best_lambda"])


def _cycle(loader):
    while True:
        yield from loader


def _prediction_rows(task: str, pred: dict, raw_validation, label_names, expert_names) -> list:
    fields = TASKS[task].text_fields
    position = {int(i): p for p, i in enumerate(raw_validation["idx"])}
    rows = []
    for j, idx in enumerate(pred["idx"]):
        example = raw_validation[position[int(idx)]]
        label, guess = int(pred["labels"][j]), int(pred["preds"][j])
        rows.append({
            "task": task,
            "idx": int(idx),
            "text": {f: example[f] for f in fields},
            "label": label,
            "label_name": label_names[label],
            "pred": guess,
            "pred_name": label_names[guess],
            "correct": label == guess,
            "probs": [round(float(x), 5) for x in pred["probs"][j]],
            "experts": list(expert_names),
            "routing": np.round(pred["routing"][j], 5).tolist(),  # [layer][expert]
        })
    return rows


def train_wemoe(
    tasks: Sequence[str],
    data_cfg: DataConfig,
    cfg: FSFTConfig,
    checkpoint_dir,
    results_dir,
    device: torch.device,
) -> dict:
    """Build WEMoE from the experts, train its routers with FSFT, evaluate once on validation."""
    from transformers import get_linear_schedule_with_warmup

    tasks = list(tasks)
    run_name = cfg.resolve_run_name()
    ckpt_dir, out_dir = run_dirs(checkpoint_dir, results_dir, run_name)
    check_expert_data_config(results_dir, tasks, data_cfg)
    init_lambda = resolve_init_lambda(cfg, results_dir)
    set_seed(cfg.seed)

    tokenizer = load_tokenizer(data_cfg.model_name)
    collator = Collator(tokenizer.pad_token_id)
    data = {t: load_task(t, tokenizer, data_cfg, splits=("fewshot_pool", "dev", "validation")) for t in tasks}
    fewshot_loaders, fewshot_idx = {}, {}
    for i, task in enumerate(tasks):
        pool = data[task]["fewshot_pool"]
        positions = sample_fewshot_indices(len(pool), cfg.k, seed=task_seed(cfg.seed, task))
        subset = pool.select(positions)
        fewshot_idx[task] = [int(x) for x in subset["idx"]]
        fewshot_loaders[task] = make_loader(subset, collator, min(cfg.batch_size, cfg.k), shuffle=True,
                                            seed=cfg.seed * 1000 + i)
    dev_loaders = {t: make_loader(d["dev"], collator, cfg.eval_batch_size) for t, d in data.items()}
    val_loaders = {t: make_loader(d["validation"], collator, cfg.eval_batch_size) for t, d in data.items()}
    amp = cfg.fp16 and device.type == "cuda"

    wemoe_cfg = WEMoEConfig(tasks=tasks, init_lambda=init_lambda, moe_scope=cfg.moe_scope, routing=cfg.routing,
                            router_hidden_layers=cfg.router_hidden_layers, impl=cfg.impl)
    pretrained = load_pretrained(data_cfg.model_name)
    experts = load_experts(checkpoint_dir, tasks)
    model = build_wemoe(pretrained, experts, wemoe_cfg)
    del pretrained, experts
    model.to(device)

    params = model.router_parameters()
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg.warmup_ratio * cfg.max_steps), cfg.max_steps)
    scaler = make_grad_scaler(amp)

    def evaluate_dev():
        per_task = evaluate_tasks(model, dev_loaders, device, amp=amp)
        return per_task, mean_score(per_task)

    start = time.time()
    dev, dev_mean = evaluate_dev()
    init_validation = evaluate_tasks(model, val_loaders, device, amp=amp)  # ~ Task Arithmetic at init_lambda
    history = [{"step": 0, "dev": dev, "dev_mean_score": dev_mean}]
    best = {"step": 0, "dev": dev, "dev_mean_score": dev_mean}
    best_state = model.router_state_dict()
    logger.info("step 0 (init): dev mean score %.4f", dev_mean)

    iterators = {t: _cycle(loader) for t, loader in fewshot_loaders.items()}
    progress = tqdm(range(1, cfg.max_steps + 1), desc=f"FSFT {run_name}")
    for step in progress:
        model.train(cfg.backbone_dropout)
        optimizer.zero_grad(set_to_none=True)
        losses = {}
        for task in tasks:  # one backward per task keeps a single task's graph in memory
            batch = next(iterators[task])
            with autocast(device, amp):
                logits = model(**model_inputs(batch, device), task=task)
            loss = F.cross_entropy(logits.float(), batch["labels"].to(device))
            scaler.scale(loss).backward()
            losses[task] = loss.item()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm or float("inf")).item()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        record = {"step": step, "loss": sum(losses.values()), "task_losses": losses, "grad_norm": grad_norm,
                  "lr": scheduler.get_last_lr()[0]}
        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            dev, dev_mean = evaluate_dev()
            record.update(dev=dev, dev_mean_score=dev_mean)
            if dev_mean > best["dev_mean_score"]:
                best = {"step": step, "dev": dev, "dev_mean_score": dev_mean}
                best_state = model.router_state_dict()
            logger.info("step %d: loss %.4f, dev mean score %.4f (best %.4f @ %d)",
                        step, record["loss"], dev_mean, best["dev_mean_score"], best["step"])
        history.append(record)
        progress.set_postfix(loss=f"{record['loss']:.3f}", best=f"{best['dev_mean_score']:.3f}")

    model.load_router_state_dict(best_state)
    predictions = {t: predict(model, val_loaders[t], t, device, amp=amp, collect_routing=True) for t in tasks}
    validation = {t: compute_metrics(t, p["preds"], p["labels"]) for t, p in predictions.items()}

    single_val = None
    if baselines_path(results_dir).exists():
        single_val = load_json(baselines_path(results_dir))["single_task"]["validation"]

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, ckpt_dir / "router.pt")
    save_json({"model_name": data_cfg.model_name, "experts_dir": str(Path(checkpoint_dir) / "experts"),
               "wemoe": wemoe_cfg.to_dict()}, ckpt_dir / "wemoe_config.json")

    results = {
        "run_name": run_name,
        "tasks": tasks,
        "data_config": data_cfg.to_dict(),
        "fsft_config": asdict(cfg),
        "wemoe_config": wemoe_cfg.to_dict(),
        "init_lambda": init_lambda,
        "fewshot_idx": fewshot_idx,
        "best_step": best["step"],
        "dev": best["dev"],
        "dev_mean_score": best["dev_mean_score"],
        "init_validation": init_validation,
        "init_validation_mean_score": mean_score(init_validation),
        "validation": validation,
        "validation_mean_score": mean_score(validation),
        "normalized": normalized_scores(validation, single_val) if single_val else None,
        "parameters": {"stored": stored_numel(model), "trainable": trainable_numel(model),
                       "breakdown": parameter_breakdown(model)},
        "train_seconds": round(time.time() - start, 1),
        "router_checkpoint": str(ckpt_dir / "router.pt"),
        "versions": library_versions(),
    }
    save_json(results, out_dir / "results.json")
    write_jsonl(history, out_dir / "history.jsonl")
    rows = []
    for task in tasks:
        rows.extend(_prediction_rows(task, predictions[task], data[task].raw_validation, data[task].label_names, tasks))
    write_jsonl(rows, out_dir / "predictions_validation.jsonl")
    logger.info("%s: validation mean score %.4f (init %.4f), best step %d; results in %s", run_name,
                results["validation_mean_score"], results["init_validation_mean_score"], best["step"], out_dir)
    return results


def load_trained_wemoe(checkpoint_dir, run_name: str, device: torch.device = torch.device("cpu")):
    """Rebuild a trained WEMoE model from its experts and the saved router weights."""
    ckpt_dir = Path(checkpoint_dir) / "wemoe" / run_name
    meta = load_json(ckpt_dir / "wemoe_config.json")
    config = WEMoEConfig(**meta["wemoe"])
    model = build_wemoe(load_pretrained(meta["model_name"]), load_experts(checkpoint_dir, config.tasks), config)
    model.load_router_state_dict(torch.load(ckpt_dir / "router.pt", map_location="cpu"))
    return model.to(device).eval()

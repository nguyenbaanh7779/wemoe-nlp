"""Single-task experts: full fine-tuning of BERT on each GLUE task, plus loading helpers."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from .data import Collator, DataConfig, load_task, make_loader, model_inputs
from .inference import evaluate_tasks
from .utils import autocast, library_versions, load_json, make_grad_scaler, save_json, set_seed

logger = logging.getLogger(__name__)


@dataclass
class ExpertTrainConfig:
    lr: float = 2e-5
    batch_size: int = 32
    eval_batch_size: int = 64
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: Optional[float] = 1.0
    fp16: bool = True
    seed: int = 42


def expert_dir(checkpoint_dir, task: str) -> Path:
    return Path(checkpoint_dir) / "experts" / task


def expert_results_path(results_dir, task: str) -> Path:
    return Path(results_dir) / "experts" / f"{task}.json"


def _quiet_transformers() -> None:
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()


def load_tokenizer(name_or_path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name_or_path)


def load_pretrained(model_name: str):
    """Pre-trained BERT (theta_0) as a sequence classifier; its random classifier is never used."""
    from transformers import AutoModelForSequenceClassification

    _quiet_transformers()
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).eval()


def load_expert(checkpoint_dir, task: str):
    from transformers import AutoModelForSequenceClassification

    path = expert_dir(checkpoint_dir, task)
    if not (path / "config.json").exists():
        raise FileNotFoundError(f"No expert for {task!r} in {path}; run scripts/train_expert.py first")
    _quiet_transformers()
    return AutoModelForSequenceClassification.from_pretrained(path).eval()


def load_experts(checkpoint_dir, tasks: Sequence[str]) -> dict:
    return {task: load_expert(checkpoint_dir, task) for task in tasks}


def check_expert_data_config(results_dir, tasks: Sequence[str], data_cfg: DataConfig) -> None:
    """Fail if an expert used a different split: the router could then see the expert's training data."""
    current = data_cfg.to_dict()
    for task in tasks:
        path = expert_results_path(results_dir, task)
        if not path.exists():
            logger.warning("No results file for expert %s (%s); cannot verify its data split", task, path)
            continue
        trained = load_json(path)["data_config"]
        diff = {k: (trained.get(k), current[k]) for k in DataConfig.SPLIT_KEYS if trained.get(k) != current[k]}
        if diff:
            raise ValueError(f"Expert {task!r} was trained with a different data split (trained, now): {diff}")


class HFClassifierAdapter(nn.Module):
    """Gives a Hugging Face sequence classifier the ``model(**inputs, task=...)`` interface."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, *, task: str):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits


def _param_groups(model: nn.Module, weight_decay: float) -> list:
    """No weight decay on biases and LayerNorm weights (all 1-D tensors)."""
    decay, no_decay = [], []
    for param in model.parameters():
        if param.requires_grad:
            (decay if param.ndim >= 2 else no_decay).append(param)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def train_expert(
    task: str,
    data_cfg: DataConfig,
    cfg: ExpertTrainConfig,
    checkpoint_dir,
    results_dir,
    device: torch.device,
) -> dict:
    """Fine-tune BERT on ``expert_train``, keep the best epoch on ``dev``, report on validation."""
    from transformers import get_linear_schedule_with_warmup

    set_seed(cfg.seed)
    tokenizer = load_tokenizer(data_cfg.model_name)
    data = load_task(task, tokenizer, data_cfg, splits=("expert_train", "dev", "validation"))
    collator = Collator(tokenizer.pad_token_id)
    train_loader = make_loader(data["expert_train"], collator, cfg.batch_size, shuffle=True, seed=cfg.seed)
    dev_loader = make_loader(data["dev"], collator, cfg.eval_batch_size)
    val_loader = make_loader(data["validation"], collator, cfg.eval_batch_size)

    model = load_pretrained(data_cfg.model_name).to(device)
    adapter = HFClassifierAdapter(model)
    use_amp = cfg.fp16 and device.type == "cuda"
    optimizer = torch.optim.AdamW(_param_groups(model, cfg.weight_decay), lr=cfg.lr)
    total_steps = len(train_loader) * cfg.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg.warmup_ratio * total_steps), total_steps)
    scaler = make_grad_scaler(use_amp)

    history, best, best_state = [], None, None
    start = time.time()
    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        running = 0.0
        progress = tqdm(train_loader, desc=f"{task} epoch {epoch}/{cfg.num_epochs}", leave=False)
        for step, batch in enumerate(progress, 1):
            with autocast(device, use_amp):
                logits = model(**model_inputs(batch, device)).logits
            loss = F.cross_entropy(logits.float(), batch["labels"].to(device))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.max_grad_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            progress.set_postfix(loss=f"{running / step:.4f}")

        dev = evaluate_tasks(adapter, {task: dev_loader}, device, amp=use_amp)[task]
        history.append({"epoch": epoch, "train_loss": running / max(1, len(train_loader)), "dev": dev})
        logger.info("%s epoch %d: train_loss=%.4f dev=%s", task, epoch, history[-1]["train_loss"], dev)
        if best is None or dev["score"] > best["dev"]["score"]:
            best = history[-1]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    validation = evaluate_tasks(adapter, {task: val_loader}, device, amp=use_amp)[task]
    out_dir = expert_dir(checkpoint_dir, task)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    results = {
        "task": task,
        "data_config": data_cfg.to_dict(),
        "train_config": asdict(cfg),
        "split_sizes": {name: len(ds) for name, ds in data.splits.items()},
        "label_names": list(data.label_names),
        "best_epoch": best["epoch"],
        "dev": best["dev"],
        "validation": validation,
        "history": history,
        "train_seconds": round(time.time() - start, 1),
        "checkpoint": str(out_dir),
        "versions": library_versions(),
    }
    save_json(results, expert_results_path(results_dir, task))
    logger.info("%s: best epoch %d, validation %s -> %s", task, best["epoch"], validation, out_dir)
    return results

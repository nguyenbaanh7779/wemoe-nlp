"""Batched prediction and evaluation shared by experts, merging baselines and WEMoE.

Every model is called as ``model(input_ids, attention_mask, token_type_ids, task=...)`` and
returns logits.
"""

from __future__ import annotations

from typing import Mapping

import torch

from .data import model_inputs
from .metrics import compute_metrics
from .utils import autocast


@torch.no_grad()
def predict(model, loader, task: str, device: torch.device, *, amp: bool = False,
            collect_routing: bool = False) -> dict:
    """Predictions for one task. ``routing`` is ``(n, num_layers, num_experts)`` for WEMoE models."""
    model.eval()
    idx, labels, preds, probs, routing = [], [], [], [], []
    for batch in loader:
        with autocast(device, amp):
            logits = model(**model_inputs(batch, device), task=task)
        p = logits.float().softmax(dim=-1).cpu()
        probs.append(p)
        preds.append(p.argmax(dim=-1))
        labels.append(batch["labels"])
        idx.append(batch["idx"])
        if collect_routing:
            routing.append(model.routing_weights().float().cpu())
    return {
        "idx": torch.cat(idx).numpy(),
        "labels": torch.cat(labels).numpy(),
        "preds": torch.cat(preds).numpy(),
        "probs": torch.cat(probs).numpy(),
        "routing": torch.cat(routing).numpy() if collect_routing else None,
    }


def evaluate_tasks(model, loaders: Mapping[str, object], device: torch.device, *, amp: bool = False) -> dict:
    """GLUE metrics per task (see ``metrics.compute_metrics``)."""
    results = {}
    for task, loader in loaders.items():
        out = predict(model, loader, task, device, amp=amp)
        results[task] = compute_metrics(task, out["preds"], out["labels"])
    return results

"""Experiment 4: stored parameters, checkpoint size and latency of the compared deployments.

Counts only depend on tensor shapes, so by default the experts are copies of the pre-trained
model; pass trained experts to measure the real checkpoints.
"""

from __future__ import annotations

import copy
import gc
import time
from typing import Mapping, Optional, Sequence

import torch

from .merging import build_task_arithmetic_model, expert_as_multihead
from .utils import autocast, stored_bytes, stored_numel, trainable_numel
from .wemoe import WEMoEConfig, build_wemoe, expected_stored_numel, parameter_breakdown


@torch.no_grad()
def measure_latency(model, task: str, device: torch.device, *, vocab_size: int, seq_len: int = 128,
                    batch_sizes: Sequence[int] = (1, 32), warmup: int = 3, iters: int = 10,
                    amp: bool = False) -> dict:
    model.to(device).eval()
    out = {}
    for batch_size in batch_sizes:
        generator = torch.Generator().manual_seed(0)
        inputs = {
            "input_ids": torch.randint(1, vocab_size, (batch_size, seq_len), generator=generator).to(device),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long, device=device),
            "token_type_ids": torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
        }
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(warmup):
            with autocast(device, amp):
                model(**inputs, task=task)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(iters):
            with autocast(device, amp):
                model(**inputs, task=task)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ms = (time.perf_counter() - start) * 1000 / iters
        row = {"ms_per_batch": round(ms, 2), "ms_per_example": round(ms / batch_size, 3)}
        if device.type == "cuda":
            row["peak_memory_mb"] = round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
        out[f"batch_{batch_size}"] = row
    model.cpu()
    return out


def efficiency_report(pretrained, tasks: Sequence[str], experts: Optional[Mapping[str, torch.nn.Module]] = None, *,
                      latency: bool = False, device: torch.device = torch.device("cpu"), seq_len: int = 128,
                      batch_sizes: Sequence[int] = (1, 32), amp: bool = False) -> dict:
    tasks = list(tasks)
    if experts is None:
        experts = {t: copy.deepcopy(pretrained) for t in tasks}
    bert_config = pretrained.config
    single = experts[tasks[0]]
    n = len(tasks)
    single_params = stored_numel(single)
    report = {
        "tasks": tasks,
        "single_task_models": {
            "num_checkpoints": n,
            "stored_params": single_params * n,
            "stored_bytes": stored_bytes(single) * n,
            "params_per_model": single_params,
            "trainable_params_finetuning": trainable_numel(single) * n,
        },
    }
    lat_kwargs = dict(vocab_size=bert_config.vocab_size, seq_len=seq_len, batch_sizes=batch_sizes, amp=amp)
    if latency:
        report["single_task_models"]["latency"] = measure_latency(
            expert_as_multihead(tasks[0], single), tasks[0], device, **lat_kwargs)

    ta = build_task_arithmetic_model(pretrained, experts, 0.3)
    report["task_arithmetic"] = {"num_checkpoints": 1, "stored_params": stored_numel(ta),
                                 "stored_bytes": stored_bytes(ta), "trainable_params": 0}
    if latency:
        report["task_arithmetic"]["latency"] = measure_latency(ta, tasks[0], device, **lat_kwargs)
    del ta

    for scope in ("full", "intermediate"):
        model = build_wemoe(pretrained, experts, WEMoEConfig(tasks=tasks, init_lambda=0.3, moe_scope=scope))
        row = {
            "num_checkpoints": 1,
            "stored_params": stored_numel(model),
            "stored_bytes": stored_bytes(model),
            "expected_params": expected_stored_numel(bert_config, n, scope),
            "trainable_params": trainable_numel(model),
            "breakdown": parameter_breakdown(model),
        }
        if latency:
            row["latency"] = measure_latency(model, tasks[0], device, **lat_kwargs)
        report[f"wemoe_{scope}"] = row
        del model
        gc.collect()

    reference = report["single_task_models"]["stored_params"]
    for key in ("single_task_models", "task_arithmetic", "wemoe_full", "wemoe_intermediate"):
        report[key]["fraction_of_single_task_models"] = round(report[key]["stored_params"] / reference, 4)
    return report

"""GLUE metrics for the five binary tasks."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from .data import TASKS


def task_score(task: str, metrics: Mapping[str, float]) -> float:
    """Single number per task: MCC for CoLA, mean(acc, F1) for MRPC, accuracy otherwise."""
    kind = TASKS[task].metric
    if kind == "acc_f1":
        return (metrics["accuracy"] + metrics["f1"]) / 2
    if kind == "mcc":
        return metrics["mcc"]
    return metrics["accuracy"]


def compute_metrics(task: str, preds, labels) -> dict:
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    metrics = {"accuracy": float(accuracy_score(labels, preds))}
    if task == "mrpc":
        metrics["f1"] = float(f1_score(labels, preds, pos_label=1, zero_division=0))
    if task == "cola":
        metrics["mcc"] = float(matthews_corrcoef(labels, preds))
    metrics["score"] = float(task_score(task, metrics))
    metrics["n"] = int(len(labels))
    return metrics


def mean_score(per_task: Mapping[str, Mapping[str, float]]) -> float:
    return float(np.mean([m["score"] for m in per_task.values()]))


def normalized_scores(
    merged: Mapping[str, Mapping[str, float]], single: Mapping[str, Mapping[str, float]]
) -> dict:
    """Score of a merged model divided by the score of the task's own expert (1.0 = no loss)."""
    out = {t: merged[t]["score"] / single[t]["score"] for t in merged if t in single and single[t]["score"]}
    if out:
        out["mean"] = float(np.mean(list(out.values())))
    return out

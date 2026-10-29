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


def majority_metrics(task: str, labels) -> dict:
    """Metrics of always predicting the most frequent label: the floor a collapsed model falls to."""
    labels = np.asarray(labels)
    return compute_metrics(task, np.full_like(labels, np.bincount(labels).argmax()), labels)


def gain_over_majority(
    merged: Mapping[str, Mapping[str, float]],
    single: Mapping[str, Mapping[str, float]],
    majority: Mapping[str, Mapping[str, float]],
) -> dict:
    """(score - majority) / (expert - majority): 0 = no better than a constant guess, 1 = the expert.

    Unlike ``normalized_scores`` it does not credit a model for the chance level of easy-to-guess
    tasks (a constant prediction already gets 0.75 on MRPC and ~0.5 on the accuracy tasks).
    """
    out = {}
    for task in merged:
        if task in single and task in majority:
            floor = majority[task]["score"]
            if single[task]["score"] > floor:
                out[task] = (merged[task]["score"] - floor) / (single[task]["score"] - floor)
    if out:
        out["mean"] = float(np.mean(list(out.values())))
    return out

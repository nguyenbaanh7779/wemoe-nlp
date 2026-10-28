"""Task vectors, static merging baselines and the multi-head classifier used for evaluation.

Simple Averaging is Task Arithmetic with ``scaling = 1 / num_tasks``:
``theta_0 + (1/T) * sum_t (theta_t - theta_0) = mean_t theta_t``.
"""

from __future__ import annotations

import copy
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

StateDict = Mapping[str, Tensor]


def task_vector(finetuned: StateDict, pretrained: StateDict) -> dict:
    """``tau = theta_ft - theta_0`` for floating-point entries. Returns new tensors."""
    if set(finetuned) != set(pretrained):
        diff = sorted(set(finetuned) ^ set(pretrained))
        raise KeyError(f"State dicts have different keys, e.g. {diff[:5]}")
    return {k: finetuned[k].detach() - v.detach() for k, v in pretrained.items() if torch.is_floating_point(v)}


def sum_task_vectors(pretrained: StateDict, finetuned: Sequence[StateDict]) -> dict:
    """``sum_t (theta_t - theta_0)``, accumulated one model at a time to keep memory low."""
    total = {}
    for state in finetuned:
        for k, tau in task_vector(state, pretrained).items():
            if k in total:
                total[k] += tau
            else:
                total[k] = tau
    return total


def add_scaled(pretrained: StateDict, delta: StateDict, scaling: float) -> dict:
    """``theta_0 + scaling * delta`` without modifying the inputs; non-float entries are copied."""
    return {
        k: v.detach() + scaling * delta[k] if k in delta else v.detach().clone() for k, v in pretrained.items()
    }


def task_arithmetic(pretrained: StateDict, task_vectors: Sequence[StateDict], scaling: float) -> dict:
    """Task Arithmetic merge ``theta_0 + scaling * sum_t tau_t`` (Ilharco et al., 2023)."""
    total = {}
    for tv in task_vectors:
        for k, tau in tv.items():
            total[k] = total[k] + tau if k in total else tau.clone()
    return add_scaled(pretrained, total, scaling)


def classification_head(model: nn.Module) -> nn.Module:
    """Copy of the dropout + linear classifier of a ``BertForSequenceClassification``."""
    return nn.Sequential(copy.deepcopy(model.dropout), copy.deepcopy(model.classifier))


class MultiHeadClassifier(nn.Module):
    """One BERT encoder shared by task heads: ``logits = heads[task](bert(x).pooler_output)``."""

    def __init__(self, bert: nn.Module, heads: Mapping[str, nn.Module]):
        super().__init__()
        self.bert = bert
        self.heads = nn.ModuleDict(dict(heads))

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, *, task: str) -> Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return self.heads[task](outputs.pooler_output)


def expert_as_multihead(task: str, expert: nn.Module) -> MultiHeadClassifier:
    """Wrap a fine-tuned expert so it shares the evaluation interface of merged models."""
    return MultiHeadClassifier(expert.bert, {task: classification_head(expert)}).eval()


def build_task_arithmetic_model(
    pretrained: nn.Module, experts: Mapping[str, nn.Module], scaling: float
) -> MultiHeadClassifier:
    return TaskArithmeticMerger(pretrained, experts).apply(scaling)


class TaskArithmeticMerger:
    """Merged encoder that can be re-merged for many lambdas without recomputing task vectors."""

    def __init__(self, pretrained: nn.Module, experts: Mapping[str, nn.Module]):
        base = pretrained.bert.state_dict()
        self.base = {k: v.detach().clone() for k, v in base.items()}
        self.delta = sum_task_vectors(self.base, [e.bert.state_dict() for e in experts.values()])
        heads = {task: classification_head(expert) for task, expert in experts.items()}
        self.model = MultiHeadClassifier(copy.deepcopy(pretrained.bert), heads)
        self.model.requires_grad_(False)
        self.model.eval()

    def to(self, device) -> "TaskArithmeticMerger":
        self.model.to(device)
        self.base = {k: v.to(device) for k, v in self.base.items()}
        self.delta = {k: v.to(device) for k, v in self.delta.items()}
        return self

    @torch.no_grad()
    def apply(self, scaling: float) -> MultiHeadClassifier:
        self.model.bert.load_state_dict(add_scaled(self.base, self.delta, scaling))
        return self.model

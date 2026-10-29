"""GLUE data: task registry, deterministic train splits, tokenisation and loaders.

The official GLUE train split of every task is divided once (seeded) into three disjoint parts:

* ``fewshot_pool`` - labelled examples reserved for FSFT router training,
* ``dev``          - model selection (expert epochs, Task Arithmetic lambda, router early stopping),
* ``expert_train`` - everything else, used to fine-tune the single-task experts.

The official GLUE validation split is only used for the numbers that are reported.
"""

from __future__ import annotations

import dataclasses
import zlib
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

GLUE_DATASET = "nyu-mll/glue"
MODEL_INPUTS = ("input_ids", "attention_mask", "token_type_ids")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    text_fields: tuple
    label_names: tuple  # same order as the HF ClassLabel; the dataset's own names are preferred
    metric: str  # "accuracy", "mcc" or "acc_f1" (mean of accuracy and F1, GLUE convention)


TASKS = {
    "mrpc": TaskSpec("mrpc", ("sentence1", "sentence2"), ("not_equivalent", "equivalent"), "acc_f1"),
    "rte": TaskSpec("rte", ("sentence1", "sentence2"), ("entailment", "not_entailment"), "accuracy"),
    "cola": TaskSpec("cola", ("sentence",), ("unacceptable", "acceptable"), "mcc"),
    "sst2": TaskSpec("sst2", ("sentence",), ("negative", "positive"), "accuracy"),
    "qnli": TaskSpec("qnli", ("question", "sentence"), ("entailment", "not_entailment"), "accuracy"),
}
DEFAULT_TASKS = ("mrpc", "rte", "cola", "sst2", "qnli")


@dataclass
class DataConfig:
    model_name: str = "bert-base-uncased"
    max_length: int = 128
    split_seed: int = 42
    fewshot_pool_size: int = 256
    dev_fraction: float = 0.1
    dev_max: int = 1000
    max_train_examples: Optional[int] = None  # smoke runs only
    max_eval_examples: Optional[int] = None  # smoke runs only

    # Keys that decide which examples end up in which split; they must match between the
    # expert, baseline and FSFT runs or the router could be trained on expert training data.
    SPLIT_KEYS = ("split_seed", "fewshot_pool_size", "dev_fraction", "dev_max", "model_name")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def split_train_indices(
    n: int, seed: int, fewshot_pool_size: int, dev_fraction: float, dev_max: int
) -> dict:
    """Disjoint, seeded split of ``range(n)`` into few-shot pool, dev and expert-train indices."""
    if fewshot_pool_size < 0 or not 0.0 <= dev_fraction < 1.0 or dev_max < 0:
        raise ValueError("Invalid split sizes")
    n_dev = min(int(round(n * dev_fraction)), dev_max)
    if fewshot_pool_size + n_dev >= n:
        raise ValueError(f"Train split of size {n} is too small for pool={fewshot_pool_size}, dev={n_dev}")
    perm = np.random.default_rng(seed).permutation(n)
    return {
        "fewshot_pool": np.sort(perm[:fewshot_pool_size]),
        "dev": np.sort(perm[fewshot_pool_size : fewshot_pool_size + n_dev]),
        "expert_train": np.sort(perm[fewshot_pool_size + n_dev :]),
    }


def task_seed(seed: int, task: str) -> list:
    """Stable per-task seed material (Python's ``hash`` is salted per process)."""
    return [seed, zlib.crc32(task.encode())]


def sample_fewshot_indices(pool_size: int, k: int, seed: int) -> np.ndarray:
    """``k`` positions in the few-shot pool. For a fixed seed, smaller k gives a subset of larger k."""
    if not 0 < k <= pool_size:
        raise ValueError(f"k={k} must be in [1, {pool_size}]")
    return np.random.default_rng(seed).permutation(pool_size)[:k]


@dataclass
class TaskData:
    task: str
    label_names: tuple
    splits: dict  # split name -> tokenised datasets.Dataset
    raw_validation: object  # untokenised official validation split (text for prediction dumps)

    def __getitem__(self, split: str):
        return self.splits[split]


def load_task(
    task: str,
    tokenizer,
    cfg: DataConfig,
    splits: Sequence[str] = ("expert_train", "dev", "fewshot_pool", "validation"),
) -> TaskData:
    """Load one GLUE task, split the official train set and tokenise the requested splits."""
    from datasets import load_dataset

    spec = TASKS[task]
    raw = load_dataset(GLUE_DATASET, task)
    label_names = tuple(raw["train"].features["label"].names)
    if label_names != spec.label_names:
        raise ValueError(f"Unexpected label names for {task}: {label_names}")

    index_splits = split_train_indices(
        len(raw["train"]),
        task_seed(cfg.split_seed, task),
        cfg.fewshot_pool_size,
        cfg.dev_fraction,
        cfg.dev_max,
    )
    raw_validation = raw["validation"]
    if cfg.max_eval_examples:
        raw_validation = raw_validation.select(range(min(cfg.max_eval_examples, len(raw_validation))))

    def tokenize(batch):
        encoded = tokenizer(
            *[batch[field] for field in spec.text_fields], truncation=True, max_length=cfg.max_length
        )
        encoded["labels"] = batch["label"]
        return encoded

    tokenized = {}
    for split in splits:
        if split == "validation":
            subset = raw_validation
        else:
            indices = index_splits[split]
            limit = cfg.max_train_examples if split == "expert_train" else cfg.max_eval_examples
            if limit and split != "fewshot_pool":
                indices = indices[:limit]
            subset = raw["train"].select(indices)
        keep = {"idx"}
        tokenized[split] = subset.map(
            tokenize,
            batched=True,
            remove_columns=[c for c in subset.column_names if c not in keep],
            desc=f"tokenize {task}/{split}",
        )
    return TaskData(task=task, label_names=label_names, splits=tokenized, raw_validation=raw_validation)


class Collator:
    """Pads model inputs to the longest sequence in the batch and keeps labels and GLUE idx."""

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, features: Sequence[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)

        def pad(key: str, value: int) -> torch.Tensor:
            return torch.tensor(
                [list(f[key]) + [value] * (max_len - len(f[key])) for f in features], dtype=torch.long
            )

        batch = {"input_ids": pad("input_ids", self.pad_token_id), "attention_mask": pad("attention_mask", 0)}
        if "token_type_ids" in features[0]:
            batch["token_type_ids"] = pad("token_type_ids", 0)
        batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        batch["idx"] = torch.tensor([f["idx"] for f in features], dtype=torch.long)
        return batch


def make_loader(dataset, collator: Collator, batch_size: int, shuffle: bool = False, seed: int = 0) -> DataLoader:
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator, generator=generator)


def model_inputs(batch: dict, device: torch.device) -> dict:
    return {k: batch[k].to(device) for k in MODEL_INPUTS if k in batch}

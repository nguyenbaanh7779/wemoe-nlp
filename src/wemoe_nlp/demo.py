"""Backend of the interactive demo (``app/demo.py``): load the models once, predict single inputs.

For one input, the demo compares three models that serve the same five tasks:

* the single-task expert of each task,
* Task Arithmetic (one statically merged BERT, lambda from ``results/baselines.json``),
* WEMoE (full MLP, ``full-mean-k256-s0`` by default), with the routing weights it used for this input.

The task only selects the classification head: the WEMoE routers see the text alone, so the routing
weights of an input are the same for every task. When no task is given, the input is scored for every
task that takes the same number of texts (one sentence: CoLA, SST-2; a pair: MRPC, RTE, QNLI).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch

from .data import DEFAULT_TASKS, MODEL_INPUTS, TASKS
from .merging import build_task_arithmetic_model, expert_as_multihead
from .utils import load_json
from .wemoe import WEMoEConfig, build_wemoe

MODEL_KEYS = ("expert", "task_arithmetic", "wemoe")
DEFAULT_RUN = "full-mean-k256-s0"  # the main result of the report: full MLP, k = 256, seed 0


@dataclass
class DemoResult:
    tasks: tuple  # tasks the input was scored for
    probs: dict  # task -> model key -> list of class probabilities
    routing: np.ndarray  # (num_layers, num_experts) WEMoE routing weights for this input
    expert_names: tuple

    def label_names(self, task: str) -> tuple:
        return TASKS[task].label_names

    def predicted_label(self, task: str, model: str) -> str:
        return self.label_names(task)[int(np.argmax(self.probs[task][model]))]


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DemoBackend:
    """Holds the experts, Task Arithmetic and one WEMoE model, all built once at start-up."""

    def __init__(
        self,
        pretrained,
        experts: Mapping[str, object],
        tokenizer,
        task_arithmetic_lambda: float,
        wemoe_config: WEMoEConfig,
        router_state: Mapping[str, torch.Tensor],
        device: Optional[torch.device] = None,
        max_length: int = 128,
    ):
        if tuple(wemoe_config.tasks) != tuple(experts):
            raise ValueError(f"WEMoE was trained on tasks {wemoe_config.tasks}, the experts are {list(experts)}")
        self.device = device or torch.device("cpu")
        self.tasks = tuple(experts)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_arithmetic_lambda = float(task_arithmetic_lambda)
        pretrained = pretrained.to(self.device).eval()
        experts = {task: model.to(self.device).eval() for task, model in experts.items()}
        self.experts = {task: expert_as_multihead(task, model) for task, model in experts.items()}
        self.task_arithmetic = build_task_arithmetic_model(pretrained, experts, self.task_arithmetic_lambda).eval()
        self.wemoe = build_wemoe(pretrained, experts, wemoe_config)
        self.wemoe.load_router_state_dict(router_state)
        self.wemoe.to(self.device).eval()

    @classmethod
    def from_checkpoints(cls, checkpoint_dir="checkpoints", results_dir="results", run: str = DEFAULT_RUN,
                         tasks: Sequence[str] = DEFAULT_TASKS, device: Optional[torch.device] = None) -> "DemoBackend":
        from .experts import expert_dir, load_experts, load_pretrained, load_tokenizer
        from .fsft import load_router

        baselines = Path(results_dir) / "baselines.json"
        if not baselines.exists():
            raise FileNotFoundError(f"{baselines} not found; it holds the Task Arithmetic lambda")
        if not (Path(checkpoint_dir) / "wemoe" / run / "router.pt").exists():
            raise FileNotFoundError(f"No router for {run} in {Path(checkpoint_dir) / 'wemoe' / run}")
        lam = load_json(baselines)["task_arithmetic"]["best_lambda"]
        model_name, config, state = load_router(checkpoint_dir, run)
        tasks = list(tasks)
        return cls(load_pretrained(model_name), load_experts(checkpoint_dir, tasks),
                   load_tokenizer(expert_dir(checkpoint_dir, tasks[0])), lam, config, state,
                   device=device or default_device())

    def tasks_for(self, num_texts: int) -> tuple:
        """Tasks that take ``num_texts`` texts: one sentence or a sentence pair."""
        return tuple(t for t in self.tasks if len(TASKS[t].text_fields) == num_texts)

    def encode(self, texts: Sequence[str]) -> dict:
        if not 1 <= len(texts) <= 2 or any(not str(t).strip() for t in texts):
            raise ValueError("Enter one sentence, or a pair of sentences")
        encoded = self.tokenizer(*[str(t) for t in texts], truncation=True, max_length=self.max_length,
                                 return_tensors="pt")
        return {k: encoded[k].to(self.device) for k in MODEL_INPUTS if k in encoded}

    @torch.no_grad()
    def predict(self, texts: Sequence[str], tasks: Optional[Sequence[str]] = None) -> DemoResult:
        """Scores ``texts`` for ``tasks`` (default: every task that takes this many texts)."""
        texts = list(texts)
        tasks = tuple(tasks) if tasks else self.tasks_for(len(texts))
        for task in tasks:
            if task not in self.experts:
                raise KeyError(f"Unknown task {task!r}; the demo serves {list(self.tasks)}")
            fields = TASKS[task].text_fields
            if len(fields) != len(texts):
                raise ValueError(f"{task} takes {len(fields)} text(s): {', '.join(fields)}")
        inputs = self.encode(texts)
        probs, routing = {}, None
        for task in tasks:
            models = {"expert": self.experts[task], "task_arithmetic": self.task_arithmetic, "wemoe": self.wemoe}
            probs[task] = {key: models[key](**inputs, task=task).float().softmax(dim=-1)[0].cpu().tolist()
                           for key in MODEL_KEYS}
            if routing is None:  # WEMoE ran last; its routing does not depend on the task
                routing = self.wemoe.routing_weights()[0].float().cpu().numpy()
        return DemoResult(tasks=tasks, probs=probs, routing=routing, expert_names=tuple(self.wemoe.wemoe_config.tasks))

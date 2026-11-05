"""Backend of the interactive demo (``app/demo.py``): load the models once, predict single inputs.

For one input, the demo compares three models that serve the same five tasks:

* the single-task expert of the selected task,
* Task Arithmetic (one statically merged BERT, lambda from ``results/baselines.json``),
* a trained WEMoE run, together with the routing weights it used for this input.

The task only selects the classification head; the WEMoE routers never see it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import torch

from .data import DEFAULT_TASKS, MODEL_INPUTS, TASKS
from .merging import build_task_arithmetic_model, expert_as_multihead
from .utils import load_json
from .wemoe import WEMoEConfig, build_wemoe

MODEL_KEYS = ("expert", "task_arithmetic", "wemoe")

# Runs offered by the demo (seed 0 of each configuration in the report), with their labels.
MAIN_RUNS = {
    "full-mean-k256-s0": "WEMoE, full MLP (main result)",
    "intermediate-mean-k256-s0": "WEMoE, intermediate.dense only",
    "full-mean-h0-k256-s0": "Ablation: layer-wise coefficients (h=0)",
}

# run name -> (WEMoEConfig, router state dict)
RunLoader = Callable[[str], "tuple[WEMoEConfig, dict]"]


@dataclass
class DemoResult:
    task: str
    label_names: tuple
    probs: dict  # model key -> list of class probabilities
    routing: np.ndarray  # (num_layers, num_experts) WEMoE routing weights for this input
    expert_names: tuple

    def predicted_label(self, model: str) -> str:
        return self.label_names[int(np.argmax(self.probs[model]))]


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def available_runs(checkpoint_dir) -> list:
    """Names of the WEMoE runs that have saved routers, in sorted order."""
    base = Path(checkpoint_dir) / "wemoe"
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if (p / "router.pt").exists() and (p / "wemoe_config.json").exists())


def demo_runs(checkpoint_dir) -> dict:
    """The ``MAIN_RUNS`` that have checkpoints, in ``MAIN_RUNS`` order: run name -> label."""
    present = set(available_runs(checkpoint_dir))
    return {name: label for name, label in MAIN_RUNS.items() if name in present}


class DemoBackend:
    """Holds the pre-trained BERT, the experts and Task Arithmetic; builds one WEMoE run at a time.

    Only the most recently used WEMoE run stays in memory (a full-MLP model is about 1.6 GB in fp32).
    """

    def __init__(
        self,
        pretrained,
        experts: Mapping[str, object],
        tokenizer,
        task_arithmetic_lambda: float,
        run_loader: RunLoader,
        device: Optional[torch.device] = None,
        max_length: int = 128,
    ):
        self.device = device or torch.device("cpu")
        self.tasks = tuple(experts)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_arithmetic_lambda = float(task_arithmetic_lambda)
        self._run_loader = run_loader
        self._pretrained = pretrained.to(self.device).eval()
        self._experts = {task: model.to(self.device).eval() for task, model in experts.items()}
        self.experts = {task: expert_as_multihead(task, model) for task, model in self._experts.items()}
        self.task_arithmetic = build_task_arithmetic_model(
            self._pretrained, self._experts, self.task_arithmetic_lambda).eval()
        self._wemoe_name: Optional[str] = None
        self._wemoe = None

    @classmethod
    def from_checkpoints(cls, checkpoint_dir="checkpoints", results_dir="results", tasks: Sequence[str] = DEFAULT_TASKS,
                         device: Optional[torch.device] = None) -> "DemoBackend":
        from .experts import expert_dir, load_experts, load_pretrained, load_tokenizer
        from .fsft import load_router

        baselines = Path(results_dir) / "baselines.json"
        if not baselines.exists():
            raise FileNotFoundError(f"{baselines} not found; it holds the Task Arithmetic lambda")
        lam = load_json(baselines)["task_arithmetic"]["best_lambda"]
        runs = available_runs(checkpoint_dir)
        if not runs:
            raise FileNotFoundError(f"No trained WEMoE run in {Path(checkpoint_dir) / 'wemoe'}")
        model_name = load_router(checkpoint_dir, runs[0])[0]
        experts = load_experts(checkpoint_dir, tasks)

        def run_loader(run_name: str):
            _, config, state = load_router(checkpoint_dir, run_name)
            if tuple(config.tasks) != tuple(tasks):
                raise ValueError(f"Run {run_name} was trained on tasks {config.tasks}, the demo serves {list(tasks)}")
            return config, state

        return cls(load_pretrained(model_name), experts, load_tokenizer(expert_dir(checkpoint_dir, tasks[0])), lam,
                   run_loader, device=device or default_device())

    def wemoe(self, run_name: str):
        """The WEMoE model of ``run_name``, built on first use (replaces the previously loaded run)."""
        if run_name != self._wemoe_name:
            config, state = self._run_loader(run_name)
            self._wemoe = self._wemoe_name = None  # free the old model before building the new one
            model = build_wemoe(self._pretrained, self._experts, config)
            model.load_router_state_dict(state)
            self._wemoe, self._wemoe_name = model.to(self.device).eval(), run_name
        return self._wemoe

    def encode(self, task: str, texts: Sequence[str]) -> dict:
        fields = TASKS[task].text_fields
        if len(texts) != len(fields):
            raise ValueError(f"{task} takes {len(fields)} text(s): {', '.join(fields)}")
        if any(not str(t).strip() for t in texts):
            raise ValueError("Every text field must be filled in")
        encoded = self.tokenizer(*[str(t) for t in texts], truncation=True, max_length=self.max_length,
                                 return_tensors="pt")
        return {k: encoded[k].to(self.device) for k in MODEL_INPUTS if k in encoded}

    @torch.no_grad()
    def predict(self, task: str, texts: Sequence[str], run_name: str) -> DemoResult:
        if task not in self.experts:
            raise KeyError(f"Unknown task {task!r}; the demo serves {list(self.tasks)}")
        inputs = self.encode(task, texts)
        wemoe = self.wemoe(run_name)
        models = {"expert": self.experts[task], "task_arithmetic": self.task_arithmetic, "wemoe": wemoe}
        probs = {}
        for key in MODEL_KEYS:
            logits = models[key](**inputs, task=task)
            probs[key] = logits.float().softmax(dim=-1)[0].cpu().tolist()
        routing = wemoe.routing_weights()[0].float().cpu().numpy()  # read right after the WEMoE forward
        return DemoResult(task=task, label_names=TASKS[task].label_names, probs=probs, routing=routing,
                          expert_names=tuple(wemoe.wemoe_config.tasks))

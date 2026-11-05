import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import TINY_TASKS, make_models
from wemoe_nlp.data import DEFAULT_TASKS, TASKS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from examples import EXAMPLES, example_title  # noqa: E402
from wemoe_nlp.demo import MAIN_RUNS, DemoBackend, available_runs, demo_runs
from wemoe_nlp.wemoe import WEMoEConfig, build_wemoe


class StubTokenizer:
    """Maps characters to ids < 50 (the tiny vocabulary); pairs get token_type_ids 0 / 1."""

    def __call__(self, *texts, truncation=True, max_length=16, return_tensors="pt"):
        ids, types = [], []
        for segment, text in enumerate(texts):
            piece = [1 + ord(c) % 48 for c in text][: max_length // len(texts)]
            ids += piece
            types += [segment] * len(piece)
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                "token_type_ids": torch.tensor([types])}


def _router_state(pretrained, experts, seed):
    """Router weights of a 'trained' run: random, so routing depends on the input."""
    model = build_wemoe(pretrained, experts, WEMoEConfig(tasks=list(experts), init_lambda=0.3))
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in model.router_parameters():
            param.copy_(torch.randn(param.shape, generator=generator) * 0.3)
    return model.router_state_dict()


@pytest.fixture
def backend():
    pretrained, experts = make_models()
    runs = {"run-a": _router_state(pretrained, experts, 0), "run-b": _router_state(pretrained, experts, 1)}
    calls = []

    def loader(name):
        calls.append(name)
        return WEMoEConfig(tasks=list(experts), init_lambda=0.3), runs[name]

    demo = DemoBackend(pretrained, experts, StubTokenizer(), 0.3, loader, max_length=16)
    demo.loader_calls = calls
    return demo


def test_predict_returns_probabilities_and_routing(backend):
    result = backend.predict("mrpc", ["the cat sat", "a cat was sitting"], "run-a")
    assert set(result.probs) == {"expert", "task_arithmetic", "wemoe"}
    for probs in result.probs.values():
        assert len(probs) == 2 and abs(sum(probs) - 1) < 1e-5
    assert result.routing.shape == (2, len(TINY_TASKS))  # (layers, experts)
    assert result.expert_names == TINY_TASKS
    assert result.predicted_label("wemoe") in result.label_names


def test_routing_depends_on_the_input(backend):
    first = backend.predict("cola", ["the cat sat"], "run-a").routing
    second = backend.predict("cola", ["zzzz qqqq xxxx"], "run-a").routing
    assert not np.allclose(first, second)


def test_only_the_last_run_stays_loaded(backend):
    backend.predict("cola", ["abc"], "run-a")
    backend.predict("cola", ["abc"], "run-a")
    backend.predict("cola", ["abc"], "run-b")
    assert backend.loader_calls == ["run-a", "run-b"]


def test_wrong_number_of_texts_is_rejected(backend):
    with pytest.raises(ValueError):
        backend.predict("mrpc", ["only one sentence"], "run-a")
    with pytest.raises(ValueError):
        backend.predict("cola", ["   "], "run-a")


def test_available_runs_needs_router_and_config(tmp_path):
    for name, files in {"ok": ("router.pt", "wemoe_config.json"), "partial": ("wemoe_config.json",)}.items():
        (tmp_path / "wemoe" / name).mkdir(parents=True)
        for f in files:
            (tmp_path / "wemoe" / name / f).write_text("x")
    assert available_runs(tmp_path) == ["ok"]
    assert available_runs(tmp_path / "missing") == []


def test_examples_match_the_task_specs():
    assert set(EXAMPLES) == set(DEFAULT_TASKS)
    for task, examples in EXAMPLES.items():
        assert examples
        for example in examples:
            assert len(example["texts"]) == len(TASKS[task].text_fields)
            assert all(text.strip() for text in example["texts"])
            assert example["label"] in TASKS[task].label_names
            assert example_title(example).startswith(f"#{example['idx']}:")


def test_demo_runs_keeps_main_runs_with_checkpoints(tmp_path):
    present = ["full-mean-h0-k256-s0", "full-mean-k64-s1", "full-mean-k256-s0"]
    for name in present:
        (tmp_path / "wemoe" / name).mkdir(parents=True)
        (tmp_path / "wemoe" / name / "router.pt").write_text("x")
        (tmp_path / "wemoe" / name / "wemoe_config.json").write_text("x")
    runs = demo_runs(tmp_path)
    assert list(runs) == ["full-mean-k256-s0", "full-mean-h0-k256-s0"]  # MAIN_RUNS order, k64 left out
    assert runs["full-mean-k256-s0"] == MAIN_RUNS["full-mean-k256-s0"]

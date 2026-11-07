import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import TINY_TASKS, make_models
from wemoe_nlp.data import DEFAULT_TASKS, TASKS
from wemoe_nlp.demo import DemoBackend
from wemoe_nlp.wemoe import WEMoEConfig, build_wemoe

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from examples import EXAMPLES, example_title  # noqa: E402


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


def _trained_router(pretrained, experts, config):
    """Router weights of a 'trained' run: random, so routing depends on the input."""
    model = build_wemoe(pretrained, experts, config)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for param in model.router_parameters():
            param.copy_(torch.randn(param.shape, generator=generator) * 0.3)
    return model.router_state_dict()


@pytest.fixture
def backend():
    pretrained, experts = make_models()  # mrpc and rte take a pair, cola one sentence
    config = WEMoEConfig(tasks=list(experts), init_lambda=0.3)
    state = _trained_router(pretrained, experts, config)
    return DemoBackend(pretrained, experts, StubTokenizer(), 0.3, config, state, max_length=16)


def test_auto_scores_every_task_with_the_same_input_format(backend):
    pair = backend.predict(["the cat sat", "a cat was sitting"])
    assert pair.tasks == ("mrpc", "rte")
    single = backend.predict(["the cat sat"])
    assert single.tasks == ("cola",)
    for result in (pair, single):
        for task in result.tasks:
            assert set(result.probs[task]) == {"expert", "task_arithmetic", "wemoe"}
            for probs in result.probs[task].values():
                assert len(probs) == 2 and abs(sum(probs) - 1) < 1e-5
            assert result.predicted_label(task, "wemoe") in TASKS[task].label_names
        assert result.routing.shape == (2, len(TINY_TASKS))  # (layers, experts)
        assert result.expert_names == TINY_TASKS


def test_selected_task_only(backend):
    result = backend.predict(["the cat sat", "a cat was sitting"], ["rte"])
    assert result.tasks == ("rte",) and set(result.probs) == {"rte"}


def test_routing_depends_on_the_input_not_on_the_task(backend):
    texts = ["the cat sat", "a cat was sitting"]
    mrpc = backend.predict(texts, ["mrpc"]).routing
    rte = backend.predict(texts, ["rte"]).routing
    np.testing.assert_allclose(mrpc, rte)
    other = backend.predict(["zzzz qqqq xxxx", "a cat was sitting"], ["mrpc"]).routing
    assert not np.allclose(mrpc, other)


def test_invalid_inputs_are_rejected(backend):
    with pytest.raises(ValueError):
        backend.predict(["only one sentence"], ["mrpc"])
    with pytest.raises(ValueError):
        backend.predict(["   "])
    with pytest.raises(ValueError):
        backend.predict(["a", "b", "c"])
    with pytest.raises(KeyError):
        backend.predict(["a"], ["sst2"])


def test_wemoe_tasks_must_match_the_experts():
    pretrained, experts = make_models()
    config = WEMoEConfig(tasks=list(experts), init_lambda=0.3)
    state = _trained_router(pretrained, experts, config)
    reordered = WEMoEConfig(tasks=list(reversed(list(experts))), init_lambda=0.3)
    with pytest.raises(ValueError):
        DemoBackend(pretrained, experts, StubTokenizer(), 0.3, reordered, state)


def test_examples_match_the_task_specs():
    assert set(EXAMPLES) == set(DEFAULT_TASKS)
    for task, examples in EXAMPLES.items():
        assert examples
        for example in examples:
            assert len(example["texts"]) == len(TASKS[task].text_fields)
            assert all(text.strip() for text in example["texts"])
            assert example["label"] in TASKS[task].label_names
            assert f"#{example['idx']}:" in example_title(example)

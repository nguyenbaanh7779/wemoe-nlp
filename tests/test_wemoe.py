import pytest
import torch
import torch.nn.functional as F

from wemoe_nlp.fsft import FSFTConfig
from wemoe_nlp.merging import build_task_arithmetic_model
from wemoe_nlp.utils import stored_numel, trainable_numel
from wemoe_nlp.wemoe import (WEMoEConfig, WeightEnsembledLinear, build_wemoe, expected_stored_numel,
                             parameter_breakdown)


def _config(experts, **kwargs):
    return WEMoEConfig(tasks=list(experts), init_lambda=0.3, **kwargs)


def _randomize_routers(model, std=0.3, seed=0):
    """Make routing clearly input-dependent (the real init is ~constant)."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in model.router_parameters():
            param.copy_(torch.randn(param.shape, generator=generator) * std)


@pytest.mark.parametrize("impl", ["merged", "decomposed"])
def test_weight_ensembled_linear_matches_formula(impl):
    torch.manual_seed(0)
    base = torch.nn.Linear(6, 5)
    experts = [torch.nn.Linear(6, 5) for _ in range(3)]
    layer = WeightEnsembledLinear(base, experts, impl="loop")
    x, w = torch.randn(4, 7, 6), torch.randn(4, 3)
    with torch.no_grad():
        reference = layer(x, w)
        for b in range(4):
            weight = base.weight + sum(w[b, e] * (experts[e].weight - base.weight) for e in range(3))
            bias = base.bias + sum(w[b, e] * (experts[e].bias - base.bias) for e in range(3))
            torch.testing.assert_close(reference[b], x[b] @ weight.T + bias)
        layer.impl = impl
        torch.testing.assert_close(layer(x, w), reference, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("scope", ["full", "intermediate"])
@pytest.mark.parametrize("routing", ["mean", "cls"])
def test_fast_implementations_match_per_sample_loop(tiny_models, tiny_batch, scope, routing):
    pretrained, experts = tiny_models
    model = build_wemoe(pretrained, experts, _config(experts, moe_scope=scope, routing=routing, impl="loop")).eval()
    _randomize_routers(model)
    with torch.no_grad():
        reference = model(**tiny_batch, task="rte")
        reference_routing = model.routing_weights()
        for impl in ("merged", "decomposed"):
            model.set_impl(impl)
            torch.testing.assert_close(model(**tiny_batch, task="rte"), reference, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(model.routing_weights(), reference_routing)


@pytest.mark.parametrize("scope", ["full", "intermediate"])
def test_constant_routing_equals_task_arithmetic(tiny_models, tiny_batch, scope):
    pretrained, experts = tiny_models
    model = build_wemoe(pretrained, experts, _config(experts, moe_scope=scope)).eval()
    with torch.no_grad():
        for layer in model.bert.encoder.layer:
            layer.intermediate.router.fc2.weight.zero_()  # routing weights become exactly init_lambda
    task_arithmetic = build_task_arithmetic_model(pretrained, experts, 0.3)
    with torch.no_grad():
        for task in experts:
            torch.testing.assert_close(model(**tiny_batch, task=task), task_arithmetic(**tiny_batch, task=task),
                                       rtol=1e-4, atol=1e-5)
        routing = model.routing_weights()
    assert routing.shape == (3, 2, len(experts))
    torch.testing.assert_close(routing, torch.full_like(routing, 0.3))


def test_masked_mean_routing_ignores_padding(tiny_models, tiny_batch):
    pretrained, experts = tiny_models
    model = build_wemoe(pretrained, experts, _config(experts, routing="mean")).eval()
    _randomize_routers(model)
    padded = {k: F.pad(v, (0, 5)) for k, v in tiny_batch.items()}  # five more padding positions
    with torch.no_grad():
        logits, routing = model(**tiny_batch, task="cola"), model.routing_weights()
        logits_padded, routing_padded = model(**padded, task="cola"), model.routing_weights()
    torch.testing.assert_close(routing_padded, routing, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(logits_padded, logits, rtol=1e-4, atol=1e-5)


def test_only_routers_train_and_receive_gradients(tiny_models, tiny_batch):
    pretrained, experts = tiny_models
    model = build_wemoe(pretrained, experts, _config(experts))
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(".router." in name for name in trainable)

    model.train(False)
    loss = F.cross_entropy(model(**tiny_batch, task="mrpc"), torch.tensor([0, 1, 1]))
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
        else:
            assert param.grad is None, name
    assert sum(p.grad.abs().sum() for p in model.router_parameters()) > 0


def test_build_does_not_modify_inputs(tiny_models):
    pretrained, experts = tiny_models
    before = {k: v.clone() for k, v in pretrained.state_dict().items()}
    before_expert = {k: v.clone() for k, v in experts["rte"].state_dict().items()}
    build_wemoe(pretrained, experts, _config(experts))
    for k, v in pretrained.state_dict().items():
        assert torch.equal(v, before[k]), k
    for k, v in experts["rte"].state_dict().items():
        assert torch.equal(v, before_expert[k]), k


@pytest.mark.parametrize("scope", ["full", "intermediate"])
@pytest.mark.parametrize("router_layers", [0, 1, 2])
def test_parameter_count_matches_closed_form(tiny_models, scope, router_layers):
    pretrained, experts = tiny_models
    model = build_wemoe(pretrained, experts, _config(experts, moe_scope=scope, router_hidden_layers=router_layers))
    expected = expected_stored_numel(pretrained.config, len(experts), scope, router_layers)
    assert stored_numel(model) == expected
    assert parameter_breakdown(model)["total"] == expected
    num_experts, hidden = len(experts), pretrained.config.hidden_size
    per_router = {0: num_experts, 1: hidden * num_experts + num_experts,
                  2: hidden * hidden + hidden + hidden * num_experts + num_experts}[router_layers]
    assert trainable_numel(model) == pretrained.config.num_hidden_layers * per_router


def test_bert_base_counts_match_review_estimates():
    """Experiment 4 numbers for BERT-base with five experts (no download: config only)."""
    from transformers import BertConfig

    config = BertConfig()  # bert-base-uncased shapes
    assert expected_stored_numel(config, 5, "full") == 399_969_094
    assert expected_stored_numel(config, 5, "intermediate") == 258_365_254


def test_router_state_roundtrip(tiny_models, tiny_batch):
    pretrained, experts = tiny_models
    trained = build_wemoe(pretrained, experts, _config(experts)).eval()
    _randomize_routers(trained)
    fresh = build_wemoe(pretrained, experts, _config(experts)).eval()
    fresh.load_router_state_dict(trained.router_state_dict())
    with torch.no_grad():
        torch.testing.assert_close(fresh(**tiny_batch, task="mrpc"), trained(**tiny_batch, task="mrpc"))
    with pytest.raises(KeyError):
        fresh.load_router_state_dict({})


def test_run_name_marks_non_default_router_depth():
    assert FSFTConfig(k=256, seed=0).resolve_run_name() == "full-mean-k256-s0"
    assert FSFTConfig(k=256, seed=0, router_hidden_layers=0).resolve_run_name() == "full-mean-h0-k256-s0"

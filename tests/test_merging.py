import copy

import torch

from wemoe_nlp.merging import (TaskArithmeticMerger, add_scaled, build_task_arithmetic_model, sum_task_vectors,
                               task_arithmetic, task_vector)


def _floats(state):
    return {k: v for k, v in state.items() if torch.is_floating_point(v)}


def test_task_arithmetic_matches_formula_and_keeps_inputs(tiny_models):
    pretrained, experts = tiny_models
    base = pretrained.bert.state_dict()
    snapshot = {k: v.clone() for k, v in base.items()}
    expert_states = [e.bert.state_dict() for e in experts.values()]

    merged = task_arithmetic(base, [task_vector(s, base) for s in expert_states], 0.3)

    for k, v in _floats(snapshot).items():
        torch.testing.assert_close(merged[k], v + 0.3 * sum(s[k] - v for s in expert_states))
    for k, v in pretrained.bert.state_dict().items():
        torch.testing.assert_close(v, snapshot[k])  # theta_0 is untouched


def test_incremental_sum_matches_task_arithmetic(tiny_models):
    pretrained, experts = tiny_models
    base = pretrained.bert.state_dict()
    states = [e.bert.state_dict() for e in experts.values()]
    expected = task_arithmetic(base, [task_vector(s, base) for s in states], 0.7)
    actual = add_scaled(base, sum_task_vectors(base, states), 0.7)
    for k in expected:
        torch.testing.assert_close(actual[k], expected[k])


def test_simple_average_is_mean_of_experts(tiny_models):
    pretrained, experts = tiny_models
    base = pretrained.bert.state_dict()
    states = [e.bert.state_dict() for e in experts.values()]
    merged = add_scaled(base, sum_task_vectors(base, states), 1 / len(states))
    for k in _floats(base):
        torch.testing.assert_close(merged[k], sum(s[k] for s in states) / len(states))


def test_legacy_danc_merge_is_an_ema_not_task_arithmetic(tiny_models):
    """Regression for WEMoETrainer.setup in the DANC code: ``sd[name] = base_sd[name]`` followed by
    ``sd[name] += ...`` updates base_sd in place, so each expert is diffed against a moving base."""
    pretrained, experts = tiny_models
    lam = 0.5
    model = copy.deepcopy(pretrained.bert)
    sd, base_sd = {}, model.state_dict()
    for name in base_sd:
        sd[name] = base_sd[name]
    for expert in experts.values():
        expert_sd = expert.bert.state_dict()
        for name in expert_sd:
            if torch.is_floating_point(expert_sd[name]) and expert_sd[name].shape == base_sd[name].shape:
                sd[name] += (expert_sd[name] - base_sd[name]) * lam

    base = pretrained.bert.state_dict()
    correct = task_arithmetic(base, [task_vector(e.bert.state_dict(), base) for e in experts.values()], lam)
    name = "encoder.layer.0.attention.self.query.weight"
    assert not torch.allclose(sd[name], correct[name])
    ema = base[name].clone()
    for expert in experts.values():
        ema = (1 - lam) * ema + lam * expert.bert.state_dict()[name]
    torch.testing.assert_close(sd[name], ema)


def test_merger_can_be_reapplied(tiny_models, tiny_batch):
    pretrained, experts = tiny_models
    merger = TaskArithmeticMerger(pretrained, experts)
    with torch.no_grad():
        first = merger.apply(0.3)(**tiny_batch, task="rte").clone()
        merger.apply(0.9)
        again = merger.apply(0.3)(**tiny_batch, task="rte")
        fresh = build_task_arithmetic_model(pretrained, experts, 0.3)(**tiny_batch, task="rte")
    torch.testing.assert_close(again, first)
    torch.testing.assert_close(fresh, first)

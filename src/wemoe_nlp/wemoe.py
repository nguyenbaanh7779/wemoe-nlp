"""Weight-Ensembling Mixture of Experts (WEMoE) for BERT.

Adapted from the reference implementation of Tang et al., "Merging Multi-Task Models via
Weight-Ensembling Mixture of Experts" (ICML 2024),
https://github.com/tanganke/weight-ensembling_MoE (``src/module/dict_moe.py``), and from its
``DictMoE`` port in the HUST_MoE / DANC code.

For one Transformer layer with routing weights ``w(x)`` (one weight per expert, *not* normalised by
a softmax) the MLP weights used for input ``x`` are::

    W(x) = W0 + sum_e w_e(x) * tau_e

where ``W0`` are the pre-trained MLP weights and ``tau_e = W_e - W0`` the task vectors. All
parameters outside the MLPs are merged statically with Task Arithmetic, ``theta_0 + lambda * sum
tau``. Routers are initialised so that ``w_e(x) ~= lambda``, i.e. the model starts as Task Arithmetic.

Changes compared with the DANC code:

* the whole MLP (``intermediate.dense`` and ``output.dense``) is upscaled with one router per layer,
  as in the paper; ``moe_scope="intermediate"`` keeps the old DANC behaviour (ablation);
* routing weights are averaged over non-padding tokens (``routing="mean"``, the paper averages over
  tokens); ``routing="cls"`` keeps the DANC behaviour;
* Task Arithmetic for the remaining parameters no longer aliases the pre-trained state dict;
* per-sample weight merging is batched instead of looping over samples.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .merging import add_scaled, classification_head, sum_task_vectors

MOE_SCOPES = ("full", "intermediate")
ROUTINGS = ("mean", "cls")
IMPLS = ("merged", "decomposed", "loop")


@dataclass
class WEMoEConfig:
    tasks: list
    init_lambda: float
    moe_scope: str = "full"
    routing: str = "mean"
    router_hidden_layers: int = 2
    impl: str = "merged"

    def __post_init__(self):
        self.tasks = list(self.tasks)
        if self.moe_scope not in MOE_SCOPES:
            raise ValueError(f"moe_scope must be one of {MOE_SCOPES}")
        if self.routing not in ROUTINGS:
            raise ValueError(f"routing must be one of {ROUTINGS}")
        if self.impl not in IMPLS:
            raise ValueError(f"impl must be one of {IMPLS}")
        if self.router_hidden_layers not in (0, 1, 2):
            raise ValueError("router_hidden_layers must be 0, 1 or 2")

    def to_dict(self) -> dict:
        return asdict(self)


class Router(nn.Module):
    """WEMoE router (``DictMoEGate`` in the reference code): unnormalised routing weights.

    With 2 hidden layers: ``fc2(relu(fc1(h)))``. Weights start near zero and the output bias at
    ``init_lambda``, so the initial routing weights are ~``init_lambda`` for every expert.
    """

    def __init__(self, hidden_size: int, num_experts: int, init_lambda: float, num_hidden_layers: int = 2):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        if num_hidden_layers == 0:
            self.weight = nn.Parameter(torch.full((num_experts,), float(init_lambda)))
            return
        if num_hidden_layers == 2:
            self.fc1 = nn.Linear(hidden_size, hidden_size)
            nn.init.normal_(self.fc1.weight, std=0.01)
            nn.init.zeros_(self.fc1.bias)
        else:
            self.fc1 = nn.Identity()
        self.fc2 = nn.Linear(hidden_size, num_experts)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.constant_(self.fc2.bias, float(init_lambda))

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.num_hidden_layers == 0:
            return self.weight.expand(*hidden_states.shape[:-1], -1)
        h = self.fc1(hidden_states)
        if self.num_hidden_layers == 2:
            h = F.relu(h)
        return self.fc2(h)


class WeightEnsembledLinear(nn.Module):
    """Linear layer with per-sample weights ``W0 + sum_e w[b, e] * tau_e`` (and the same for bias).

    ``impl`` picks one of three equivalent computations:

    * ``"merged"`` (default): build the per-sample weights, then one batched matmul. Fewest FLOPs.
    * ``"decomposed"``: ``x W0^T + sum_e w_e (x tau_e^T)`` as a single matmul over the stacked task
      vectors. Keeps only ``x`` for backward, so it uses the least memory when training.
    * ``"loop"``: per-sample merge in a Python loop, like the reference code. Only for tests.
    """

    def __init__(self, base: nn.Linear, experts: Sequence[nn.Linear], impl: str = "merged"):
        super().__init__()
        if impl not in IMPLS:
            raise ValueError(f"impl must be one of {IMPLS}")
        self.impl = impl
        with torch.no_grad():
            w0 = base.weight.detach().clone()
            tau = torch.stack([e.weight.detach() - w0 for e in experts])  # (E, out, in)
            self.register_buffer("weight0", w0)
            self.register_buffer("tau_weight_t", tau.transpose(1, 2).contiguous())  # (E, in, out)
            if base.bias is not None:
                b0 = base.bias.detach().clone()
                self.register_buffer("bias0", b0)
                self.register_buffer("tau_bias", torch.stack([e.bias.detach() - b0 for e in experts]))
            else:
                self.register_buffer("bias0", None)
                self.register_buffer("tau_bias", None)

    @property
    def num_experts(self) -> int:
        return self.tau_weight_t.shape[0]

    def merged_weight(self, w: Tensor) -> Tensor:
        """Per-sample weight matrices ``(B, out, in)`` for routing weights ``w`` of shape ``(B, E)``."""
        E, I, O = self.tau_weight_t.shape
        delta_t = (w @ self.tau_weight_t.reshape(E, I * O)).view(-1, I, O)
        return self.weight0.unsqueeze(0) + delta_t.transpose(1, 2)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """``x``: ``(B, L, in)``, ``w``: ``(B, E)`` -> ``(B, L, out)``."""
        E, I, O = self.tau_weight_t.shape
        if self.impl == "loop":
            outputs = []
            for b in range(x.shape[0]):
                weight = self.weight0 + torch.einsum("e,eio->oi", w[b], self.tau_weight_t)
                bias = None if self.bias0 is None else self.bias0 + w[b] @ self.tau_bias
                outputs.append(F.linear(x[b], weight, bias))
            return torch.stack(outputs)
        if self.impl == "merged":
            delta_t = (w @ self.tau_weight_t.reshape(E, I * O)).view(-1, I, O)
            # add in delta's dtype (fp16 under autocast) instead of promoting B weight copies to fp32
            y = torch.bmm(x, self.weight0.t().to(delta_t.dtype).unsqueeze(0) + delta_t)
        else:
            mixed = (x.unsqueeze(2) * w[:, None, :, None]).reshape(*x.shape[:2], E * I)
            y = F.linear(x, self.weight0) + mixed @ self.tau_weight_t.reshape(E * I, O)
        if self.bias0 is not None:
            y = y + (self.bias0 + w @ self.tau_bias).unsqueeze(1)
        return y


class RoutingContext:
    """Per-forward information shared by all routers (the attention mask for masked means)."""

    def __init__(self):
        self.attention_mask: Optional[Tensor] = None


class LayerRouting:
    """Routing weights of one layer, shared by its fc1 and fc2 and kept for analysis."""

    def __init__(self):
        self.weights: Optional[Tensor] = None


class WEMoEIntermediate(nn.Module):
    """Replaces ``BertIntermediate``: computes the layer's routing weights, then fc1 + activation."""

    def __init__(
        self,
        router: Router,
        fc1: WeightEnsembledLinear,
        activation: nn.Module,
        context: RoutingContext,
        state: LayerRouting,
        routing: str = "mean",
    ):
        super().__init__()
        self.router = router
        self.fc1 = fc1
        self.intermediate_act_fn = activation
        self.routing = routing
        self._context = context
        self._state = state

    def routing_weights(self, hidden_states: Tensor) -> Tensor:
        if self.routing == "cls":
            return self.router(hidden_states[:, 0])
        token_weights = self.router(hidden_states)  # (B, L, E)
        mask = self._context.attention_mask
        if mask is None:
            return token_weights.mean(dim=1)
        mask = mask.to(token_weights.dtype).unsqueeze(-1)
        return (token_weights * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def forward(self, hidden_states: Tensor) -> Tensor:
        weights = self.routing_weights(hidden_states)
        self._state.weights = weights
        return self.intermediate_act_fn(self.fc1(hidden_states, weights))


class WEMoEOutputDense(nn.Module):
    """Replaces ``BertOutput.dense`` (fc2), reusing the routing weights computed by the layer's fc1."""

    def __init__(self, fc2: WeightEnsembledLinear, state: LayerRouting):
        super().__init__()
        self.fc2 = fc2
        self._state = state

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.fc2(hidden_states, self._state.weights)


class WEMoEForSequenceClassification(nn.Module):
    """Merged BERT with WEMoE MLPs and one frozen classification head per task."""

    def __init__(
        self,
        bert: nn.Module,
        heads: Mapping[str, nn.Module],
        config: WEMoEConfig,
        context: RoutingContext,
        layer_states: Sequence[LayerRouting],
    ):
        super().__init__()
        self.bert = bert
        self.heads = nn.ModuleDict(dict(heads))
        self.wemoe_config = config
        self._context = context
        self._layer_states = list(layer_states)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, *, task: str) -> Tensor:
        self._context.attention_mask = attention_mask
        try:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        finally:
            self._context.attention_mask = None
        return self.heads[task](outputs.pooler_output)

    def routing_weights(self) -> Tensor:
        """Routing weights of the last forward pass: ``(batch, num_layers, num_experts)``."""
        weights = [state.weights for state in self._layer_states]
        if any(w is None for w in weights):
            raise RuntimeError("No routing weights yet: run a forward pass first")
        return torch.stack([w.detach() for w in weights], dim=1)

    def router_parameters(self) -> list:
        return [p for name, p in self.named_parameters() if ".router." in name]

    def router_state_dict(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if ".router." in k}

    def load_router_state_dict(self, state: Mapping[str, Tensor]) -> None:
        expected = {k for k in self.state_dict() if ".router." in k}
        if set(state) != expected:
            raise KeyError(f"Router state mismatch: missing={sorted(expected - set(state))[:3]}, "
                           f"unexpected={sorted(set(state) - expected)[:3]}")
        self.load_state_dict(state, strict=False)

    def set_impl(self, impl: str) -> None:
        if impl not in IMPLS:
            raise ValueError(f"impl must be one of {IMPLS}")
        for module in self.modules():
            if isinstance(module, WeightEnsembledLinear):
                module.impl = impl
        self.wemoe_config.impl = impl


@torch.no_grad()
def build_wemoe(
    pretrained: nn.Module, experts: Mapping[str, nn.Module], config: WEMoEConfig
) -> WEMoEForSequenceClassification:
    """Upscale the MLPs of the pre-trained BERT into WEMoE modules and merge the rest with TA.

    ``pretrained`` and ``experts`` are ``BertForSequenceClassification`` models; the experts are
    used in the order of ``config.tasks``. Inputs are not modified.
    """
    missing = [t for t in config.tasks if t not in experts]
    if missing:
        raise KeyError(f"Missing experts for tasks {missing}")
    base_bert = pretrained.bert
    if getattr(base_bert.config, "chunk_size_feed_forward", 0):
        raise ValueError("Feed-forward chunking would split the routing over the sequence")
    expert_berts = [experts[t].bert for t in config.tasks]

    base_state = base_bert.state_dict()
    merged = copy.deepcopy(base_bert)
    delta = sum_task_vectors(base_state, [b.state_dict() for b in expert_berts])
    merged.load_state_dict(add_scaled(base_state, delta, config.init_lambda))
    del delta

    context = RoutingContext()
    states = []
    hidden_size = base_bert.config.hidden_size
    for i, layer in enumerate(merged.encoder.layer):
        base_layer = base_bert.encoder.layer[i]
        expert_layers = [b.encoder.layer[i] for b in expert_berts]
        state = LayerRouting()
        router = Router(hidden_size, len(config.tasks), config.init_lambda, config.router_hidden_layers)
        fc1 = WeightEnsembledLinear(
            base_layer.intermediate.dense, [l.intermediate.dense for l in expert_layers], config.impl
        )
        layer.intermediate = WEMoEIntermediate(
            router, fc1, layer.intermediate.intermediate_act_fn, context, state, config.routing
        )
        if config.moe_scope == "full":
            fc2 = WeightEnsembledLinear(base_layer.output.dense, [l.output.dense for l in expert_layers], config.impl)
            layer.output.dense = WEMoEOutputDense(fc2, state)
        states.append(state)

    heads = {task: classification_head(experts[task]) for task in config.tasks}
    model = WEMoEForSequenceClassification(merged, heads, copy.deepcopy(config), context, states)
    for name, param in model.named_parameters():
        param.requires_grad_(".router." in name)
    return model


def expected_stored_numel(bert_config, num_experts: int, moe_scope: str = "full",
                          router_hidden_layers: int = 2, num_labels: int = 2) -> int:
    """Closed-form number of stored floating-point values of a WEMoE model built from BERT."""
    H, I, E = bert_config.hidden_size, bert_config.intermediate_size, num_experts
    embeddings = (bert_config.vocab_size + bert_config.max_position_embeddings + bert_config.type_vocab_size) * H + 2 * H
    attention = 4 * (H * H + H) + 2 * H
    fc1 = H * I + I
    fc2 = I * H + H
    output_layer_norm = 2 * H
    router = {0: E, 1: H * E + E, 2: H * H + H + H * E + E}[router_hidden_layers]
    per_layer = attention + fc1 * (1 + E) + (fc2 * (1 + E) if moe_scope == "full" else fc2) + output_layer_norm + router
    pooler = H * H + H
    heads = E * (H * num_labels + num_labels)
    return embeddings + bert_config.num_hidden_layers * per_layer + pooler + heads


def parameter_breakdown(model: WEMoEForSequenceClassification) -> dict:
    """Stored values per component: shared backbone, pre-trained MLP, task vectors, routers, heads."""
    counts = dict.fromkeys(("shared_backbone", "base_mlp", "task_vectors", "routers", "heads"), 0)
    for name, tensor in model.state_dict().items():
        if not torch.is_floating_point(tensor):
            continue
        if name.startswith("heads."):
            key = "heads"
        elif ".router." in name:
            key = "routers"
        elif name.endswith(("tau_weight_t", "tau_bias")):
            key = "task_vectors"
        elif name.endswith(("weight0", "bias0")):
            key = "base_mlp"
        else:
            key = "shared_backbone"
        counts[key] += tensor.numel()
    counts["total"] = sum(counts.values())
    return counts

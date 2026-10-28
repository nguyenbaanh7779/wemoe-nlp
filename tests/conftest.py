"""Tiny random BERT models so the tests run offline on CPU in seconds."""

import copy

import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification

TINY_TASKS = ("mrpc", "rte", "cola")


def tiny_config() -> BertConfig:
    return BertConfig(vocab_size=50, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                      intermediate_size=32, max_position_embeddings=32, type_vocab_size=2, num_labels=2)


def make_models(seed: int = 0, tasks=TINY_TASKS):
    """A 'pre-trained' model and experts that differ from it by random task vectors."""
    torch.manual_seed(seed)
    pretrained = BertForSequenceClassification(tiny_config()).eval()
    experts = {}
    for task in tasks:
        expert = copy.deepcopy(pretrained)
        with torch.no_grad():
            for param in expert.parameters():
                param.add_(0.05 * torch.randn_like(param))
        experts[task] = expert.eval()
    return pretrained, experts


@pytest.fixture
def tiny_models():
    return make_models()


@pytest.fixture
def tiny_batch():
    """Three sequences of different lengths, right-padded with id 0."""
    generator = torch.Generator().manual_seed(1)
    lengths = [9, 6, 4]
    max_len = max(lengths)
    mask = torch.zeros(len(lengths), max_len, dtype=torch.long)
    for i, n in enumerate(lengths):
        mask[i, :n] = 1
    input_ids = torch.randint(1, 50, (len(lengths), max_len), generator=generator) * mask
    token_type_ids = torch.zeros_like(input_ids)
    token_type_ids[:, 3:] = 1
    return {"input_ids": input_ids, "attention_mask": mask, "token_type_ids": token_type_ids * mask}

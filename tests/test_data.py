import numpy as np
import pytest

from wemoe_nlp.data import TASKS, Collator, DataConfig, sample_fewshot_indices, split_train_indices, task_seed


def test_split_is_disjoint_complete_and_deterministic():
    first = split_train_indices(2490, 42, 256, 0.1, 1000)
    second = split_train_indices(2490, 42, 256, 0.1, 1000)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])

    pool, dev, train = first["fewshot_pool"], first["dev"], first["expert_train"]
    assert (len(pool), len(dev), len(train)) == (256, 249, 2490 - 256 - 249)
    assert not set(pool) & set(dev) and not set(pool) & set(train) and not set(dev) & set(train)
    assert sorted(np.concatenate([pool, dev, train]).tolist()) == list(range(2490))
    assert not np.array_equal(pool, split_train_indices(2490, 7, 256, 0.1, 1000)["fewshot_pool"])


def test_dev_split_is_capped():
    assert len(split_train_indices(104743, 42, 256, 0.1, 1000)["dev"]) == 1000


def test_split_rejects_a_too_small_train_set():
    with pytest.raises(ValueError):
        split_train_indices(100, 0, 90, 0.1, 1000)


def test_task_seed_is_stable_and_task_specific():
    assert task_seed(42, "rte") == task_seed(42, "rte")
    assert task_seed(42, "rte") != task_seed(42, "mrpc")


def test_fewshot_samples_are_nested_and_deterministic():
    seed = task_seed(0, "rte")
    k16 = sample_fewshot_indices(256, 16, seed)
    k64 = sample_fewshot_indices(256, 64, seed)
    np.testing.assert_array_equal(k64[:16], k16)
    assert len(set(k64.tolist())) == 64 and k64.max() < 256
    with pytest.raises(ValueError):
        sample_fewshot_indices(32, 64, 0)


def test_label_names_follow_glue():
    # Hugging Face GLUE: label 0 is "entailment" for RTE and QNLI (inverted in the DANC report).
    assert TASKS["rte"].label_names[0] == "entailment"
    assert TASKS["qnli"].label_names[0] == "entailment"
    assert TASKS["cola"].label_names[1] == "acceptable"
    assert TASKS["sst2"].label_names[1] == "positive"
    assert TASKS["mrpc"].label_names[1] == "equivalent"


def test_collator_pads_inputs_and_keeps_labels():
    features = [
        {"input_ids": [101, 7, 102], "attention_mask": [1, 1, 1], "token_type_ids": [0, 0, 0], "labels": 1, "idx": 5},
        {"input_ids": [101, 102], "attention_mask": [1, 1], "token_type_ids": [0, 0], "labels": 0, "idx": 9},
    ]
    batch = Collator(pad_token_id=0)(features)
    assert batch["input_ids"].tolist() == [[101, 7, 102], [101, 102, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch["token_type_ids"].tolist() == [[0, 0, 0], [0, 0, 0]]
    assert batch["labels"].tolist() == [1, 0] and batch["idx"].tolist() == [5, 9]


def test_split_keys_are_data_config_fields():
    assert set(DataConfig.SPLIT_KEYS) <= set(DataConfig().to_dict())

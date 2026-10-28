import pytest

from wemoe_nlp.metrics import compute_metrics, mean_score, normalized_scores


def test_mrpc_reports_accuracy_f1_and_their_mean():
    metrics = compute_metrics("mrpc", [1, 1, 0, 1], [1, 0, 0, 1])
    assert metrics["accuracy"] == 0.75
    assert metrics["f1"] == pytest.approx(0.8)
    assert metrics["score"] == pytest.approx((0.75 + 0.8) / 2)


def test_cola_scores_mcc_and_constant_predictions_get_zero():
    metrics = compute_metrics("cola", [1, 1, 1, 1], [1, 0, 1, 0])
    assert metrics["mcc"] == 0.0 and metrics["score"] == 0.0


@pytest.mark.parametrize("task", ["rte", "sst2", "qnli"])
def test_accuracy_tasks(task):
    metrics = compute_metrics(task, [0, 1], [0, 0])
    assert metrics["score"] == 0.5 and set(metrics) == {"accuracy", "score", "n"}


def test_always_paraphrase_on_mrpc_gives_the_danc_tta_number():
    # 279 of the 408 MRPC validation pairs are paraphrases. Predicting "paraphrase" for all of them
    # gives acc 0.6838 / F1 0.8122, the Adapter-WEMoE TTA result in the DANC report (Table 3.7).
    metrics = compute_metrics("mrpc", [1] * 408, [1] * 279 + [0] * 129)
    assert round(metrics["accuracy"], 4) == 0.6838
    assert round(metrics["f1"], 4) == 0.8122


def test_mean_and_normalized_scores():
    single = {"rte": {"score": 0.7}, "sst2": {"score": 0.9}}
    merged = {"rte": {"score": 0.35}, "sst2": {"score": 0.9}}
    assert mean_score(merged) == pytest.approx(0.625)
    normalized = normalized_scores(merged, single)
    assert normalized["rte"] == pytest.approx(0.5) and normalized["sst2"] == pytest.approx(1.0)
    assert normalized["mean"] == pytest.approx(0.75)

import numpy as np

from src.evaluating import Evaluator


def test_evaluate_perfect() -> None:
    # Perfect predictions: top-1 predicted class matches the true label for every sample
    y_true = np.array([0, 1, 2])
    y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]])
    evaluator = Evaluator()
    metrics = evaluator.evaluate(y_true, y_pred)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_evaluate_imperfect() -> None:
    # Some incorrect predictions: only one out of three predictions is correct
    # For k=1, the correct prediction is when the argmax equals the true label
    y_true = np.array([0, 1, 2])
    y_pred = np.array(
        [
            [0.1, 0.85, 0.05],  # predicted: 1 (incorrect for true label 0)
            [0.2, 0.7, 0.1],  # predicted: 1 (correct for true label 1)
            [0.6, 0.3, 0.1],  # predicted: 0 (incorrect for true label 2)
        ]
    )
    evaluator = Evaluator()
    metrics = evaluator.evaluate(y_true, y_pred)
    expected_precision = 0.1667
    expected_recall = 0.3333
    expected_f1 = 0.2222
    assert abs(metrics["precision"] - expected_precision) < 1e-4
    assert abs(metrics["recall"] - expected_recall) < 1e-4
    assert abs(metrics["f1"] - expected_f1) < 1e-4


def test_evaluate_imperfect_2() -> None:
    # Some incorrect predictions: only one out of three predictions is correct
    # For k=1, the correct prediction is when the argmax equals the true label
    y_true = np.array([0, 1, 1, 1, 1, 0, 2, 1, 0, 1])
    y_pred = np.array(
        [
            [0.85, 0.1, 0.05],  # predicted: 0 (correct for true label 0)
            [0.2, 0.1, 0.7],  # predicted: 2 (incorrect for true label 1)
            [0.3, 0.6, 0.1],  # predicted: 1 (correct for true label 1)
            [0.2, 0.7, 0.1],  # predicted: 1 (correct for true label 1)
            [0.1, 0.1, 0.8],  # predicted: 2 (incorrect for true label 1)
            [0.1, 0.4, 0.5],  # predicted: 2 (incorrect for true label 0)
            [0.0, 0.1, 0.9],  # predicted: 2 (correct for true label 2)
            [0.45, 0.35, 0.2],  # predicted: 0 (incorrect for true label 1)
            [0.4, 0.3, 0.3],  # predicted: 0 (correct for true label 0)
            [0.1, 0.7, 0.2],  # predicted: 1 (correct for true label 1)
        ]
    )
    evaluator = Evaluator()
    metrics = evaluator.evaluate(y_true, y_pred)
    expected_precision = 0.6389
    expected_recall = 0.7222
    expected_f1 = 0.5778
    assert abs(metrics["precision"] - expected_precision) < 1e-4
    assert abs(metrics["recall"] - expected_recall) < 1e-4
    assert abs(metrics["f1"] - expected_f1) < 1e-4


def test_evaluate_none_correct() -> None:
    # No correct predictions: all predicted top-1 labels do not match the true labels
    y_true = np.array([1, 1, 1])
    y_pred = np.array(
        [
            [0.9, 0.05, 0.05],  # predicted: 0
            [0.8, 0.1, 0.1],  # predicted: 0
            [0.7, 0.15, 0.15],  # predicted: 0
        ]
    )
    evaluator = Evaluator()
    metrics = evaluator.evaluate(y_true, y_pred)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


class Evaluator:
    def _get_top_k_predictions(self, predictions: np.ndarray, k: int) -> np.ndarray:
        """
        Get the top-k predictions for each sample.

        Args:
            predictions (np.ndarray): Array of predicted probabilities for each class
                                    Shape: (n_samples, n_classes)

        Returns:
            np.ndarray: Top-k predicted class indices for each sample
                       Shape: (n_samples, k)
        """
        return np.argsort(predictions, axis=1)[:, -k:]

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """
        Evaluate predictions using multiple metrics.

        Args:
            y_true (np.ndarray): True class indices, shape (n_samples,)
            y_pred (np.ndarray): Predicted probabilities, shape (n_samples, n_classes)

        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics
        """
        y_pred_top = self._get_top_k_predictions(y_pred, k=1).squeeze()
        return {
            "precision": precision_score(
                y_true, y_pred_top, average="macro", zero_division=0
            ),
            "recall": recall_score(
                y_true, y_pred_top, average="macro", zero_division=0
            ),
            "f1": f1_score(y_true, y_pred_top, average="macro", zero_division=0),
        }

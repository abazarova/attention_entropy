import typing as tp
from abc import ABC, abstractmethod  # noqa: D100

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import f1_score

T = tp.TypeVar("T")

class HallucinationDetectionMethod(ABC):
    """Abstract class for specifying method signatures and interface of the general hallucination detection methods."""

    @abstractmethod
    def fit(
        self,
        X_train: list[T],
        y_train: pd.Series,
        X_val: tp.Optional[list[T]],
        y_val: tp.Optional[list[int]]
    ) -> "HallucinationDetectionMethod":
        """Fit the detection model on training data."""
        pass

    @abstractmethod
    def predict_score(self, X: list[T]) -> np.ndarray[float]:
        """Predict probability of hallucination for samples from X."""
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> tp.Sequence[T]:
        """Predict probability of hallucination for samples from X."""
        pass

    def reset(self):
        """For supervised models - return the model to the untrained state."""
        pass

    def fit_threshold(self, X: list[T], y: pd.Series) -> None:
        """Fit the best threshold based on the predicted scores and true labels.

        Parameters
        ----------
        X : list
            Input data.
        y : list
            True labels.

        """
        # scores = self.predict_score(X)
        # self.best_thr = self.get_threshold(scores, y)
        self.best_thr = 0.5

    def fit_hyperparameters(self, X_val: pd.DataFrame, y_val: pd.Series):
        """Perform hyperparameter tuning on the validation set if needed."""
        pass

    def predict(self, X: list[T]) -> np.ndarray[int]:
        """Perform inference and return binary predictions based on the fitted threshold.

        Parameters
        ----------
        X : list
            DataFrame containing the input data.

        Returns
        -------
        prediction : list
            Binary predictions (0 or 1) based on the threshold.

        """
        if self.best_thr is None:
            raise ValueError("Threshold is not fitted. Call 'fit_threshold' first.")

        scores = np.array(self.predict_score(X))
        prediction = np.int32(scores > self.best_thr)
        return prediction


    @staticmethod
    def get_threshold(train_scores: np.ndarray[float], train_labels: pd.Series) -> float:
        """Get the optimal threshold that maximizes F1 score on the training set.

        Parameters
        ----------
        train_scores : np.ndarray
            Predicted scores.
        train_labels : np.ndarray
            True labels.

        Returns
        -------
        float
            The best threshold that maximizes the F1 score.

        """
        max_f1 = 0
        max_thr = 0
        thresholds = np.linspace(np.min(train_scores), np.max(train_scores), 1000)
        for thr in thresholds:
            predictions = (train_scores > thr).astype(int)
            f1 = f1_score(train_labels, predictions)
            if f1 > max_f1:
                max_thr = thr
                max_f1 = f1

        logger.info(f'Selected threshold: {max_thr} with F1 score: {max_f1}')
        return max_thr

"""
Abstract base class for APO estimators.
"""

import numpy as np
from abc import ABC, abstractmethod


class APOEstimator(ABC):
    """Abstract base class for APO estimators."""

    @abstractmethod
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray,
            val_X: np.ndarray, val_T: np.ndarray, val_Y: np.ndarray,
            M: int) -> None:
        """
        Fit the estimator to observed data.

        Args:
            X: (n, p) confounders
            T: (n,) treatment assignments
            Y: (n,) observed outcomes
            val_X: (n_val, p) validation confounders
            val_T: (n_val,) validation treatment assignments
            val_Y: (n_val,) validation observed outcomes
            M: Number of treatments
        """
        pass

    @abstractmethod
    def predict_apos(self, X: np.ndarray, M: int) -> np.ndarray:
        """
        Estimate APOs using the fitted model.

        Args:
            X: (n, p) confounders (typically the training X)
            M: Number of treatments

        Returns:
            (M,) estimated APOs
        """
        pass

    def fit_predict(self, dataset) -> np.ndarray:
        """
        Convenience method to fit and predict on a dataset.

        Trains on the training set and computes APO estimates using the validation set.

        Args:
            dataset: SyntheticDataset instance

        Returns:
            (M,) estimated APOs
        """
        self.fit(dataset.X_train, dataset.T_train, dataset.Y_train,
                 dataset.X_val, dataset.T_val, dataset.Y_val, dataset.M)
        return self.predict_apos(dataset.X_val, dataset.M)

"""
Outcome Imputation estimator for APO estimation.
"""

import numpy as np

from .base import APOEstimator
from ..models.outcome_models import OutcomeModel


class OutcomeImputation(APOEstimator):
    """
    APO estimator using outcome imputation (supervised learning).

    Method:
    1. Train a model μ(x,t) to predict Y from (X,T) using observed data
    2. For each treatment t, predict μ(X_i, t) for all samples i
    3. Estimate APO(t) = (1/n) Σ_i μ(X_i, t)
    """

    def __init__(
        self,
        outcome_model: OutcomeModel,
        verbose: bool = False
    ):
        """
        Args:
            outcome_model: Outcome model instance (linear/mlp)
            verbose: Whether to print progress messages
        """
        self.outcome_model = outcome_model
        self.verbose = verbose

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray,
            val_X: np.ndarray, val_T: np.ndarray, val_Y: np.ndarray,
            M: int, T_multi: np.ndarray = None, val_T_multi: np.ndarray = None,
            all_T_multi: np.ndarray = None, feature_sizes: list = None,
            treatment_sizes: list = None, prompt_format=None, **kwargs) -> None:
        """Fit outcome model to observed data.

        Args:
            X: (n, d_x) confounders
            T: (n,) flat treatment indices (unused internally, kept for interface)
            Y: (n,) observed outcomes
            val_X: (n_val, d_x) validation confounders
            val_T: (n_val,) validation flat treatment indices
            val_Y: (n_val,) validation observed outcomes
            M: Number of treatments
            T_multi: (n, d_t) treatment feature vectors
            val_T_multi: (n_val, d_t) validation treatment feature vectors
            all_T_multi: (M, d_t) feature vectors for all M treatments (for predict_apos)
            feature_sizes: Cardinalities per X dimension
            treatment_sizes: Cardinalities per T dimension
        """
        if self.verbose:
            print("Fitting outcome model...")

        self.all_T_multi = all_T_multi
        self.outcome_model.fit(X, T_multi, Y, M,
                               val_X=val_X, val_T_multi=val_T_multi, val_Y=val_Y,
                               feature_sizes=feature_sizes,
                               treatment_sizes=treatment_sizes,
                               prompt_format=prompt_format)

    def predict_apos(self, X: np.ndarray, M: int) -> np.ndarray:
        """Estimate APOs by imputation, marginalising over the empirical X distribution.

        For each unique covariate value x, predicts μ(x, t) for all M treatments
        in one batch call, then weights by the empirical p(x). 

        Args:
            X: (n, d_x) confounders
            M: Number of treatments

        Returns:
            (M,) array of APO estimates
        """
        unique_xs, counts = np.unique(X, axis=0, return_counts=True)
        weights = counts / counts.sum()

        apo_estimates = np.zeros(M)
        for x_val, w in zip(unique_xs, weights):
            X_cf = np.tile(x_val, (M, 1))
            apo_estimates += w * self.outcome_model.predict(X_cf, self.all_T_multi, M)

        return apo_estimates

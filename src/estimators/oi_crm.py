"""
Outcome Imputation (OI) estimator for APO estimation via CRM.
"""
import numpy as np
from typing import Optional

from .base import APOEstimator
from ..models.outcome_models import OutcomeModel
from ..models.apo_models import APOModel


class OICRM(APOEstimator):
    """
    APO estimator that trains g(t) with labels derived from an outcome imputation model f(x, t).

    Method:
    1. Train outcome model f(x, t) to predict Y from (X, T) using observed data
    2. For each treatment t, compute the f-derived APO: f_apos[t] = (1/n) Σ_i f(t, X_i)
    3. Construct per-observation labels by looking up each observed treatment: label_i = f_apos[T_i]
    4. Train APO model g(t) on (T_multi, labels)
    5. Estimate APO(t) = g(t)
    """

    def __init__(
        self,
        outcome_model: OutcomeModel,
        apo_model: APOModel,
        verbose: bool = False,
    ):
        """
        Args:
            outcome_model: Outcome model instance (linear/mlp) used as f(x, t)
            apo_model: APO model instance (linear/mlp) used as g(t)
            verbose: Whether to print progress messages
        """
        self.outcome_model = outcome_model
        self.apo_model = apo_model
        self.verbose = verbose
        self.all_T_multi: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        val_X: np.ndarray,
        val_T: np.ndarray,
        val_Y: np.ndarray,
        M: int,
        T_multi: Optional[np.ndarray] = None,
        val_T_multi: Optional[np.ndarray] = None,
        all_T_multi: Optional[np.ndarray] = None,
        feature_sizes: Optional[list] = None,
        treatment_sizes: Optional[list] = None,
        prompt_format=None,
        **kwargs,
    ) -> None:
        """
        Fit the OI-CRM estimator.

        Args:
            X: (n, d_x) confounders
            T: (n,) flat treatment indices
            Y: (n,) observed outcomes
            val_X: (n_val, d_x) validation confounders
            val_T: (n_val,) validation flat treatment indices
            val_Y: (n_val,) validation observed outcomes
            M: Number of treatments
            T_multi: (n, d_t) treatment feature vectors for training observations
            val_T_multi: (n_val, d_t) validation treatment feature vectors
            all_T_multi: (M, d_t) feature vectors for all M treatments
            feature_sizes: Cardinalities per X dimension (for transformer outcome models)
            treatment_sizes: Cardinalities per T dimension
            prompt_format: Prompt format (for HF models)
        """
        self.all_T_multi = all_T_multi

        # Split training data in half: 1 for outcome model, 2 for APO model
        n = len(X)
        idx1, idx2 = np.arange(n // 2), np.arange(n // 2, n)
        X1, X2 = X[idx1], X[idx2]
        T1, T2 = T[idx1], T[idx2]
        Y1, Y2 = Y[idx1], Y[idx2]
        T_multi1 = T_multi[idx1] if T_multi is not None else None
        T_multi2 = T_multi[idx2] if T_multi is not None else None

        if self.verbose:
            print("Fitting outcome model...")
        self.outcome_model.fit(X1, T_multi1, Y1, M,
                               val_X=val_X, val_T_multi=val_T_multi, val_Y=val_Y,
                               feature_sizes=feature_sizes,
                               treatment_sizes=treatment_sizes,
                               prompt_format=prompt_format)

        if self.verbose:
            print("Computing f-derived APO labels...")
        labels = self._compute_f_apo_labels(X2, T2, M)
        val_labels = self._compute_f_apo_labels(val_X, val_T, M)

        if self.verbose:
            print("Training APO model on f-derived labels...")
        apo_kwargs = {}
        if treatment_sizes is not None:
            apo_kwargs['treatment_sizes'] = treatment_sizes
        if prompt_format is not None:
            apo_kwargs['prompt_format'] = prompt_format
        self.apo_model.fit(T_multi2, labels, M,
                           val_T_multi=val_T_multi, val_Y=val_labels,
                           **apo_kwargs)

    def _compute_f_apo_labels(self, X: np.ndarray, T: np.ndarray, M: int) -> np.ndarray:
        """
        Compute per-observation labels from the fitted outcome model.

        For each of the M treatments, marginalizes f over X to get f_apos[t] = (1/n) Σ_j f(t, X_j).
        Then returns labels[i] = f_apos[T_i] for each training observation.

        Args:
            X: (n, d_x) confounders
            T: (n,) flat treatment indices
            M: Number of treatments

        Returns:
            (n,) per-observation labels derived from f
        """
        unique_xs, counts = np.unique(X, axis=0, return_counts=True)
        weights = counts / counts.sum()

        f_apos = np.zeros(M)
        for x_val, w in zip(unique_xs, weights):
            X_cf = np.tile(x_val, (M, 1))
            f_apos += w * self.outcome_model.predict(X_cf, self.all_T_multi, M)
        return f_apos[T]

    def predict_apos(self, X: np.ndarray, M: int) -> np.ndarray:
        """
        Estimate APOs using the trained g(t) model.

        X is unused since g is a treatment-only model; the argument is kept
        for interface compatibility with APOEstimator.

        Args:
            X: (n, d_x) confounders (unused)
            M: Number of treatments

        Returns:
            (M,) array of APO estimates
        """
        return self.apo_model.predict(self.all_T_multi, M)

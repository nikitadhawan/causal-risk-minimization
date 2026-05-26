"""Stabilized Weight (SW) estimator for APO estimation via CRM.
"""

import numpy as np
from typing import Dict, Optional

from .base import APOEstimator


class SWCRM(APOEstimator):
    """
    APO estimator using Direct Importance Weight estimation.

    Instead of learning p(t|x) and computing w = p(t)/p(t|x) as in IPW,
    directly estimates stabilized weights w(x,t) via an SW model trained
    on balance losses.

    Method:
    1. Fit w_model to directly predict w(x,t) = p(t)/p(t|x)
    2. Compute w_i for each observed (X_i, T_i)
    3. Train APO model g(t) with sw-weighted labels
    4. Estimate APO(t) = g(t)
    """

    def __init__(
        self,
        w_model,
        apo_model,
        verbose: bool = False,
    ):
        """
        Args:
            w_model: SW model instance (SWModel) for direct weight estimation
            apo_model: APO model instance (APOModel)
            verbose: Whether to print progress messages
        """
        self.w_model = w_model
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
        balance_moment_targets: Optional[Dict[int, np.ndarray]] = None,
        feature_sizes: Optional[list] = None,
        treatment_sizes: Optional[list] = None,
        T_multi: Optional[np.ndarray] = None,
        val_T_multi: Optional[np.ndarray] = None,
        all_T_multi: Optional[np.ndarray] = None,
        prompt_format=None,
    ) -> None:
        """
        Fit SW-CRM estimator to observed data.

        Args:
            X: (n, d_x) confounders
            T: (n,) flat treatment indices
            Y: (n,) observed outcomes
            val_X: (n_val, d_x) validation confounders
            val_T: (n_val,) validation flat treatment indices
            val_Y: (n_val,) validation observed outcomes
            M: Number of treatments
            balance_moment_targets: Dict mapping j -> (d_x,) array of E[X^j] for j=1..k
            feature_sizes: Cardinalities per X dimension (for transformer SW models)
            treatment_sizes: Cardinalities per T dimension
            T_multi: (n, d_t) treatment feature vectors
            val_T_multi: (n_val, d_t) validation treatment feature vectors
            all_T_multi: (M, d_t) feature vectors for all M treatments
            prompt_format: Prompt format (for HF SW models)
        """
        self.all_T_multi = all_T_multi

        # Split training data in half: 1 for SW model, 2 for APO model
        n = len(X)
        idx1, idx2 = np.arange(n // 2), np.arange(n // 2, n)
        X1, X2 = X[idx1], X[idx2]
        T1, T2 = T[idx1], T[idx2]
        Y1, Y2 = Y[idx1], Y[idx2]
        T_multi1 = T_multi[idx1] if T_multi is not None else None
        T_multi2 = T_multi[idx2] if T_multi is not None else None

        if self.verbose:
            print("Fitting SW model...")

        self.w_model.fit(X1, T_multi1, T1, M,
                         val_X=val_X, val_T_multi=val_T_multi, val_T=val_T,
                         balance_moment_targets=balance_moment_targets,
                         feature_sizes=feature_sizes, treatment_sizes=treatment_sizes,
                         prompt_format=prompt_format)
        importance_weights = self.w_model.predict_weights(X2, T_multi2)
        val_importance_weights = self.w_model.predict_weights(val_X, val_T_multi)

        if self.verbose:
            print("Training APO model with importance-weighted labels...")

        apo_kwargs = {}
        if treatment_sizes is not None:
            apo_kwargs['treatment_sizes'] = treatment_sizes
        if prompt_format is not None:
            apo_kwargs['prompt_format'] = prompt_format
        self.apo_model.fit(T_multi2, Y2, M,
                           val_T_multi=val_T_multi, val_Y=val_Y,
                           weights=importance_weights,
                           val_weights=val_importance_weights,
                           **apo_kwargs)

    def predict_apos(self, X: np.ndarray, M: int) -> np.ndarray:
        """Estimate APOs using the trained g(t) model."""
        return self.apo_model.predict(self.all_T_multi, M)

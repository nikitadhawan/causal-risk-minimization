"""
Inverse Propensity Weighting (IPW) estimator for APO estimation via CRM.
"""

import numpy as np
from typing import Dict, Optional

from .base import APOEstimator
from ..utils import x_to_flat


class IPWCRM(APOEstimator):
    """
    APO estimator using Inverse Propensity Weighting.

    Method:
    1. Train propensity model p(t|x) to predict T from X
    2. Compute marginal p(t) via Monte Carlo approximation
    3. Train APO model g(t) with importance-weighted labels:
       weighted_label_i = Y_i * p(T_i) / p(T_i | X_i)
    4. Estimate APO(t) = g(t)
    """

    def __init__(
        self,
        propensity_model,
        apo_model,
        verbose: bool = False
    ):
        """
        Args:
            propensity_model: Propensity model instance (OraclePropensityModel/EmpiricalPropensityModel/LearnedPropensityModel)
            apo_model: APO model instance (APOModel)
            verbose: Whether to print progress messages
        """
        self.propensity_model = propensity_model
        self.apo_model = apo_model
        self.verbose = verbose
        self.train_propensity_table: Optional[np.ndarray] = None
        self.true_E_X: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        val_X: np.ndarray,
        val_T: np.ndarray,
        val_Y: np.ndarray,
        M: int,
        propensity_table: Optional[np.ndarray] = None,
        true_E_X: Optional[np.ndarray] = None,
        balance_moment_targets: Dict = None,
        feature_sizes: Optional[list] = None,
        treatment_sizes: Optional[list] = None,
        T_multi: Optional[np.ndarray] = None,
        val_T_multi: Optional[np.ndarray] = None,
        all_T_multi: Optional[np.ndarray] = None,
        prompt_format=None,
    ) -> None:
        """
        Fit IPW-CRM estimator to observed data.

        Args:
            X: (n, d_x) confounders
            T: (n,) flat treatment indices
            Y: (n,) observed outcomes
            val_X: (n_val, d_x) validation confounders
            val_T: (n_val,) validation flat treatment indices
            val_Y: (n_val,) validation observed outcomes
            M: Number of treatments
            propensity_table: Optional (n_x_combos, M) true propensity table for oracle mode
            true_E_X: Optional (d_x,) precomputed E[X], used for balance regularization
            feature_sizes: Cardinalities per X dimension
            treatment_sizes: Cardinalities per T dimension 
            T_multi: (n, d_t) treatment feature vectors
            val_T_multi: (n_val, d_t) validation treatment feature vectors
            all_T_multi: (M, d_t) feature vectors for all M treatments (for predict_apos)
        """
        if self.verbose:
            print("Fitting propensity model...")

        fit_kwargs = {}
        if propensity_table is not None:
            fit_kwargs["propensity_table"] = propensity_table
        if true_E_X is not None:
            fit_kwargs["true_E_X"] = true_E_X
        if balance_moment_targets is not None:
            fit_kwargs["balance_moment_targets"] = balance_moment_targets
        if feature_sizes is not None:
            fit_kwargs["feature_sizes"] = feature_sizes
        if treatment_sizes is not None:
            fit_kwargs["treatment_sizes"] = treatment_sizes
        if T_multi is not None:
            fit_kwargs["T_multi"] = T_multi
        if all_T_multi is not None:
            fit_kwargs["all_T_multi"] = all_T_multi
        if prompt_format is not None:
            fit_kwargs["prompt_format"] = prompt_format

        # Split training data in half: 1 for propensity, 2 for APO
        n = len(X)
        idx1, idx2 = np.arange(n // 2), np.arange(n // 2, n)
        X1, X2 = X[idx1], X[idx2]
        T1, T2 = T[idx1], T[idx2]
        Y1, Y2 = Y[idx1], Y[idx2]
        T_multi1 = T_multi[idx1] if T_multi is not None else None
        T_multi2 = T_multi[idx2] if T_multi is not None else None

        self.true_E_X = true_E_X
        if T_multi is not None:
            fit_kwargs["T_multi"] = T_multi1
            fit_kwargs["val_T_multi"] = val_T_multi
        fit_kwargs['val_X'] = val_X
        fit_kwargs['val_T'] = val_T
        self.propensity_model.fit(X1, T1, M, **fit_kwargs)

        if feature_sizes and not hasattr(self.propensity_model, 'marginal_proba'):
            n_x_combos = int(np.prod(feature_sizes))
            self.train_propensity_table = np.zeros((n_x_combos, M))
            train_propensity_scores = self.propensity_model.predict_proba(X1, M)
            x_flat = x_to_flat(X1, feature_sizes)
            for x in range(n_x_combos):
                mask = x_flat == x
                if mask.any():
                    self.train_propensity_table[x] = train_propensity_scores[mask].mean(axis=0)

        self.all_T_multi = all_T_multi

        if self.verbose:
            print("Training APO model with importance-weighted labels...")

        self._fit_apo_model(X2, T2, T_multi2, Y2, val_X, val_T, val_Y, M,
                            val_T_multi=val_T_multi, treatment_sizes=treatment_sizes,
                            prompt_format=prompt_format)

    def compute_marginal(self, X: np.ndarray, M: int) -> None:
        """Compute marginal p(t) using Monte Carlo approximation."""
        # Monte Carlo approximation: p(t) ≈ (1/n) Σ_i p(t | X_i)
        propensity_scores = self.propensity_model.predict_proba(X, M)
        return propensity_scores.mean(axis=0)

    def _fit_apo_model(self, X: np.ndarray, T: np.ndarray, T_multi: np.ndarray,
                       Y: np.ndarray, val_X: np.ndarray, val_T: np.ndarray,
                       val_Y: np.ndarray, M: int, val_T_multi: np.ndarray = None,
                       treatment_sizes: list = None, prompt_format=None) -> None:
        """Train APO model g(t) with importance-weighted labels."""
        if hasattr(self.propensity_model, 'marginal_proba'):
            # Generative LM interface: score specific (X_i, T_i) pairs directly.
            propensity_scores = self.propensity_model.predict_proba(X, T_multi)
            marginal_scores = self.propensity_model.marginal_proba(X, T_multi, true_E_X=self.true_E_X)
        else:
            # Matrix interface: predict_proba returns (n, M).
            propensity_scores_all = self.propensity_model.predict_proba(X, M)
            propensity_scores = propensity_scores_all[np.arange(len(T)), T]
            marginal_scores = self.compute_marginal(X, M)[T]

        importance_weights = marginal_scores / (propensity_scores + 1e-8)

        if hasattr(self.propensity_model, 'marginal_proba'):
            val_prop_scores = self.propensity_model.predict_proba(val_X, val_T_multi)
            val_marginal_scores = self.propensity_model.marginal_proba(val_X, val_T_multi, true_E_X=self.true_E_X)
        else:
            val_prop_scores_all = self.propensity_model.predict_proba(val_X, M)
            val_prop_scores = val_prop_scores_all[np.arange(len(val_T)), val_T]
            val_marginal_scores = self.compute_marginal(val_X, M)[val_T]
        val_weights = val_marginal_scores / (val_prop_scores + 1e-8)

        apo_kwargs = {}
        if treatment_sizes is not None:
            apo_kwargs['treatment_sizes'] = treatment_sizes
        if prompt_format is not None:
            apo_kwargs['prompt_format'] = prompt_format
        self.apo_model.fit(T_multi, Y, M,
                           val_T_multi=val_T_multi, val_Y=val_Y,
                           weights=importance_weights,
                           val_weights=val_weights,
                           **apo_kwargs)

    def predict_apos(self, X: np.ndarray, M: int) -> np.ndarray:
        """Estimate APOs using the trained g(t) model."""
        return self.apo_model.predict(self.all_T_multi, M)

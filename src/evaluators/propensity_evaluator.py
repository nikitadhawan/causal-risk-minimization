"""
Evaluator for propensity-score-based estimators.
"""

import numpy as np
from typing import Dict

from ..utils import x_to_flat


class PropensityEvaluator:
    """Evaluates propensity score quality and balance for IPW estimators."""

    @staticmethod
    def evaluate(
        estimator,
        dataset,
        X: np.ndarray,
        T: np.ndarray,
        T_multi: np.ndarray = None,
        k_eval: int = None,
    ) -> Dict[str, float]:
        """
        Args:
            estimator: IPW estimator with fitted propensity model
            dataset: dataset with true propensity_table (optional for LM models)
            X: (n, 1) scalar confounders
            T: (n,) flat treatment indices
            T_multi: (n, d_t) treatment feature vectors (required for LM propensity models)
            k_eval: Number of moments to evaluate. Defaults to propensity_model.balance_moments.

        Returns:
            Dictionary with avg_balance_score, avg_kl_divergence,
            marginal_kl_divergence. KL metrics are NaN for LM propensity models.
            Balance score: sum_{j=1}^{k} mean((E_w[Z^j | T=t] - E[Z^j])^2).
        """
        k = k_eval if k_eval is not None else getattr(estimator.propensity_model, 'balance_moments', 1)
        moment_targets = dataset.x_moments(k)

        if estimator.train_propensity_table is None:
            return PropensityEvaluator._evaluate_lm(estimator, X, T, T_multi, moment_targets, k)

        M = dataset.M
        x_flat = x_to_flat(X, dataset.feature_sizes)

        train_marginal_probs = estimator.train_propensity_table.mean(axis=0)
        true_marginal_probs = np.array([
            (dataset.T == t).mean() for t in range(M)
        ])
        marginal_kl = np.sum(train_marginal_probs * np.log(train_marginal_probs / true_marginal_probs))

        balance_scores = PropensityEvaluator._compute_balance_scores(
            X, T, M, x_flat, estimator.train_propensity_table, moment_targets, k,
        )
        kl_per_x = PropensityEvaluator._compute_kl_divergence(
            dataset.propensity_table, estimator.train_propensity_table,
        )

        return {
            'balance_scores': balance_scores,
            'avg_balance_score': np.mean(balance_scores),
            'kl_divergence': kl_per_x,
            'avg_kl_divergence': kl_per_x.mean(),
            'marginal_kl_divergence': marginal_kl,
        }

    @staticmethod
    def _evaluate_lm(
        estimator, X: np.ndarray, T: np.ndarray, T_multi: np.ndarray,
        moment_targets: dict, k: int,
    ) -> Dict[str, float]:
        """
        Propensity evaluation for LM models where no true propensity table exists.

        Computes aggregate balance and norm scores using importance weights
        w_i = p_marg(T_i) / p(T_i | X_i).
        """
        assert T_multi is not None, "T_multi required for LM propensity evaluation"
        true_E_X = moment_targets['mean']
        propensity_scores = estimator.propensity_model.predict_proba(X, T_multi)
        marginal_scores = estimator.propensity_model.marginal_proba(X, T_multi, true_E_X)
        weights = marginal_scores / (propensity_scores + 1e-8)

        mean = moment_targets['mean']
        std  = moment_targets['std']
        Z = (X.astype(np.float32) - mean) / std
        w_norm = weights / weights.sum()

        balance_score = (weights.mean() - 1.0) ** 2
        for j in range(1, k + 1):
            weighted_moment = (w_norm[:, None] * Z ** j).sum(axis=0)
            balance_score += ((weighted_moment - moment_targets[j]) ** 2).mean()

        return {
            'avg_balance_score': balance_score,
            'avg_kl_divergence': float('nan'),
            'marginal_kl_divergence': float('nan'),
        }

    @staticmethod
    def _compute_balance_scores(
        X: np.ndarray,
        T: np.ndarray,
        M: int,
        x_flat: np.ndarray,
        propensity_probs: np.ndarray,
        moment_targets: dict,
        k: int,
    ) -> np.ndarray:
        """
        Compute balance score for each treatment: sum_{j=0}^{k} error_j, where
          j=0: (E[w_full | T=t] - 1)^2  (norm check, w_full = p_marg / p_cond)
          j>=1: mean((E_w[Z^j | T=t] - E[Z^j])^2)  (standardized moment balance)
        """
        mean = moment_targets['mean']
        std  = moment_targets['std']
        Z = (X.astype(np.float32) - mean) / std
        p_marg = propensity_probs.mean(axis=0)

        balance_scores = np.zeros(M)
        for t in range(M):
            mask = T == t
            if mask.sum() == 0:
                continue
            Z_t = Z[mask]
            prop_t = propensity_probs[x_flat[mask], t]

            w_full = p_marg[t] / prop_t
            balance_scores[t] += (w_full.mean() - 1.0) ** 2

            w = 1.0 / prop_t
            w = w / w.sum()
            for j in range(1, k + 1):
                weighted_moment = (w[:, None] * Z_t ** j).sum(axis=0)
                balance_scores[t] += ((weighted_moment - moment_targets[j]) ** 2).mean()
        return balance_scores

    @staticmethod
    def _compute_kl_divergence(
        true_propensity_table: np.ndarray,
        estimated_propensity_table: np.ndarray,
    ) -> np.ndarray:
        """Compute KL(estimated || true) for each X combo."""
        n_x_combos = len(true_propensity_table)
        kl_per_x = np.zeros(n_x_combos)

        for x in range(n_x_combos):
            estimated_probs = estimated_propensity_table[x]
            true_probs = true_propensity_table[x]
            kl_per_x[x] = np.sum(estimated_probs * np.log(estimated_probs / true_probs))

        return kl_per_x

"""
Evaluator for stabilized-weight-based models.
"""

import numpy as np
from typing import Dict


class SWEvaluator:
    """Evaluates stabilized weight quality for SW-CRM estimators."""

    @staticmethod
    def evaluate(
        estimator,
        dataset,
        X: np.ndarray,
        T: np.ndarray,
        T_multi: np.ndarray,
        k_eval: int = None,
    ) -> Dict[str, float]:
        """
        Args:
            estimator: SWCRM estimator with fitted w_model
            dataset: dataset with x_moments(k) method
            X: (n, d_x) confounders
            T: (n,) flat treatment indices
            T_multi: (n, d_t) treatment feature vectors
            k_eval: Number of moments to evaluate. Defaults to estimator.w_model.balance_moments.
                    Override to a fixed value (e.g. always 4) for fair cross-k comparisons.

        Returns:
            Dictionary with balance_scores, avg_balance_score.
            Balance score per treatment: sum_{j=0}^{k} ((E_w[Z^j | T=t] - E[Z^j])^2)
            Lower is better.
        """
        M = dataset.M
        k = k_eval if k_eval is not None else getattr(estimator.w_model, 'balance_moments', 1)
        moment_targets = dataset.x_moments(k)
        weights = estimator.w_model.predict_weights(X, T_multi) 

        mean = moment_targets['mean']
        std  = moment_targets['std']
        Z    = (X.astype(np.float32) - mean) / std

        balance_scores = np.zeros(M)

        for t in range(M):
            mask = T == t
            if mask.sum() == 0:
                continue

            w_t = weights[mask]
            Z_t = Z[mask]

            balance_scores[t] += (w_t.mean() - 1.0) ** 2
            for j in range(1, k + 1):
                weighted_moment = (w_t[:, None] * Z_t ** j).mean(axis=0)  # (d_x,)
                balance_scores[t] += ((weighted_moment - moment_targets[j]) ** 2).mean()

        return {
            'balance_scores': balance_scores,
            'avg_balance_score': np.mean(balance_scores),
        }

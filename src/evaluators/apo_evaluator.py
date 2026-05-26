"""
Evaluator for APO estimators.
"""

import numpy as np
from scipy.stats import spearmanr
from typing import Dict

from ..models.propensity_models import OraclePropensityModel, LearnedPropensityModel
from ..models.transformer_propensity import TransformerPropensityModel


class APOEvaluator:
    """Evaluates APO estimation."""

    @staticmethod
    def evaluate(estimator, dataset) -> Dict[str, float]:
        """
        Evaluate an APO estimator on a dataset.

        The estimator is trained on the training set and APO estimates are computed
        on both training and validation sets.

        Args:
            estimator: APOEstimator instance
            dataset: Dataset instance

        Returns:
            Dictionary containing relative metrics for both train and val:
            - train_rel_mse, val_rel_mse: Relative mean squared error
            - train_rel_mae, val_rel_mae: Relative mean absolute error
            - train_rel_max_error, val_rel_max_error: Relative maximum absolute error
            - train_pearson, val_pearson: Pearson correlation with true APOs
            - train_spearman, val_spearman: Spearman correlation with true APOs
            - true_apos: (M,) true APO values
            - train_estimated_apos, val_estimated_apos: (M,) estimated APO values
        """
        fit_kwargs = {
            'T_multi': dataset.T_multi_train,
            'val_T_multi': dataset.T_multi_val,
            'all_T_multi': dataset.all_T_multi,
            'feature_sizes': dataset.feature_sizes,
            'treatment_sizes': dataset.treatment_sizes,
        }
        if hasattr(dataset, 'prompt_format'):
            fit_kwargs['prompt_format'] = dataset.prompt_format
        if hasattr(estimator, 'propensity_model'):
            if isinstance(estimator.propensity_model, OraclePropensityModel):
                if hasattr(dataset, 'propensity_table'):
                    fit_kwargs['propensity_table'] = dataset.propensity_table
            if hasattr(dataset, 'true_E_X'):
                fit_kwargs['true_E_X'] = dataset.true_E_X
            if hasattr(dataset, 'x_moments'):
                k = getattr(estimator.propensity_model, 'balance_moments', 1)
                fit_kwargs['balance_moment_targets'] = dataset.x_moments(k)
        if hasattr(estimator, 'w_model') and hasattr(dataset, 'x_moments'):
            k = getattr(estimator.w_model, 'balance_moments', 1)
            fit_kwargs['balance_moment_targets'] = dataset.x_moments(k)

        estimator.fit(dataset.X_train, dataset.T_train, dataset.Y_train,
                      dataset.X_val, dataset.T_val, dataset.Y_val,
                      dataset.M, **fit_kwargs)

        train_ts = np.unique(dataset.T_train)
        val_ts = np.unique(dataset.T_val)

        train_apo_estimates = estimator.predict_apos(dataset.X_train, dataset.M)[train_ts]
        val_apo_estimates = estimator.predict_apos(dataset.X_val, dataset.M)[val_ts]

        train_relative_errors = (train_apo_estimates - dataset.true_apos[train_ts]) / dataset.true_apos[train_ts]
        train_rel_mse = np.mean(train_relative_errors**2)
        train_rel_mae = np.mean(np.abs(train_relative_errors))
        train_rel_max_error = np.max(np.abs(train_relative_errors))
        train_pearson = np.corrcoef(dataset.true_apos[train_ts], train_apo_estimates)[0, 1] if len(train_ts) > 1 else np.nan
        train_spearman = spearmanr(dataset.true_apos[train_ts], train_apo_estimates).statistic if len(train_ts) > 1 else np.nan

        val_relative_errors = (val_apo_estimates - dataset.true_apos[val_ts]) / dataset.true_apos[val_ts]
        val_rel_mse = np.mean(val_relative_errors**2)
        val_rel_mae = np.mean(np.abs(val_relative_errors))
        val_rel_max_error = np.max(np.abs(val_relative_errors))
        val_pearson = np.corrcoef(dataset.true_apos[val_ts], val_apo_estimates)[0, 1] if len(val_ts) > 1 else np.nan
        val_spearman = spearmanr(dataset.true_apos[val_ts], val_apo_estimates).statistic if len(val_ts) > 1 else np.nan

        return {
            'train_rel_mse': train_rel_mse,
            'train_rel_mae': train_rel_mae,
            'train_rel_max_error': train_rel_max_error,
            'train_pearson': train_pearson,
            'train_spearman': train_spearman,
            'val_rel_mse': val_rel_mse,
            'val_rel_mae': val_rel_mae,
            'val_rel_max_error': val_rel_max_error,
            'val_pearson': val_pearson,
            'val_spearman': val_spearman,
            'true_apos': dataset.true_apos,
            'train_estimated_apos': train_apo_estimates,
            'val_estimated_apos': val_apo_estimates
        }

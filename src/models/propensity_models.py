"""
Propensity score models for estimating p(t | x).

All models follow a common interface:
- fit(X, T, M, **kwargs): Train the model
- predict_proba(X, M): Return (n, M) array of propensity scores
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Literal

from .linear import LinearModel
from .mlp import MLPModel
from ..utils import x_to_flat


class BalNormCriterion(nn.Module):
    """
    Balance + normalization regularization.

    For each treatment t:
      balance_loss = (1/M) * sum_k sum_t || E_w[X^k | T=t] - E[X^k] ||^2
        where w_i = 1 / p(t | X_i), self-normalized within each treatment group.
      norm_loss = (1/M) * sum_t (E[p_marg(t) / p(t | X) | T=t] - 1)^2
        penalizes deviation of mean IPW weight from 1 per treatment.

    Args:
        balance_weight: Weight on the balance regularization term
        norm_weight: Weight on the normalization regularization term
        true_E_X: (p,) precomputed E[X]
    """

    def __init__(
        self,
        balance_weight: float,
        norm_weight: float = 0.0,
        balance_moments: int = 1,
        moment_targets: Dict = None,
    ):
        super().__init__()
        self.balance_weight = balance_weight
        self.norm_weight = norm_weight
        self.balance_moments = balance_moments
        self.moment_targets = moment_targets 
        self.base_criterion = None  # injected by PropensityModel's fit() based on task

    def forward(self, logits: torch.Tensor, T: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (n, M) unnormalized propensity scores
            T: (n,) treatment labels
            X: (n, p) covariates
        """
        base_loss = (self.base_criterion(logits, T)
                     if self.base_criterion is not None
                     else torch.tensor(0.0, device=logits.device))
        balance_loss = self._balance(logits, T, X) if self.balance_weight > 0.0 else 0.0
        norm_loss = self._norm(logits, T) if self.norm_weight > 0.0 else 0.0
        return base_loss + self.balance_weight * balance_loss + self.norm_weight * norm_loss

    def _balance(self, logits: torch.Tensor, T: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)  # (n, M)
        M = logits.shape[1]
        balance_loss = torch.tensor(0.0, device=logits.device)

        mean = self.moment_targets['mean'].to(logits.device)
        std  = self.moment_targets['std'].to(logits.device)
        Z    = (X - mean) / std

        for t in range(M):
            mask = T == t
            if mask.sum() == 0:
                continue
            w = 1.0 / probs[mask, t]
            w = w / w.sum()
            for j in range(1, self.balance_moments + 1):
                Z_pow = Z[mask] ** j
                weighted_moment = (w.unsqueeze(1) * Z_pow).sum(dim=0)
                balance_loss = balance_loss + ((weighted_moment - self.moment_targets[j].to(logits.device)) ** 2).mean() / (2 ** j)

        return balance_loss

    def _norm(self, logits: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)  # (n, M)
        p_marg = probs.mean(dim=0)            # (M,) MC marginal estimate
        M = logits.shape[1]
        norm_loss = torch.tensor(0.0, device=logits.device)
        for t in range(M):
            mask = T == t
            if mask.sum() == 0:
                continue
            w = p_marg[t] / probs[mask, t]  
            norm_loss = norm_loss + (w.mean() - 1.0) ** 2
        return norm_loss


class OraclePropensityModel:
    """Oracle propensity model using true propensity table."""

    def __init__(self):
        """Initialize oracle propensity model."""
        self.propensity_table: Optional[np.ndarray] = None
        self.feature_sizes: Optional[list] = None

    def fit(self, X: np.ndarray, T: np.ndarray, M: int, **kwargs) -> None:
        """Store the true propensity table.

        Args:
            X: (n, d_x) confounders
            T: (n,) treatment assignments
            M: Number of treatments
            kwargs: Must include propensity_table: (n_x_combos, M) true propensity table;
                    and feature_sizes: list of cardinalities per X dimension
        """
        self.propensity_table = kwargs['propensity_table']
        self.feature_sizes = kwargs['feature_sizes']

    def predict_proba(self, X: np.ndarray, M: int) -> np.ndarray:
        x_flat = x_to_flat(X, self.feature_sizes) 
        return self.propensity_table[x_flat]


class EmpiricalPropensityModel:
    """Empirical propensity model using observed frequencies."""

    def __init__(self):
        """Initialize empirical propensity model."""
        self.propensity_table: Optional[np.ndarray] = None
        self.feature_sizes: Optional[list] = None

    def fit(self, X: np.ndarray, T: np.ndarray, M: int, **kwargs) -> None:
        """Compute empirical propensity frequencies p(t | x) from data.

        Args:
            X: (n, d_x) confounders
            T: (n,) treatment assignments
            M: Number of treatments
            kwargs: Must include feature_sizes: list of cardinalities per X dimension
        """
        self.feature_sizes = kwargs['feature_sizes']
        n_x_combos = int(np.prod(self.feature_sizes))
        x_flat = x_to_flat(X, self.feature_sizes)
        self.propensity_table = np.zeros((n_x_combos, M))

        for x in range(n_x_combos):
            mask = x_flat == x
            for t in range(M):
                self.propensity_table[x, t] = (T[mask] == t).mean() + 1e-8 if mask.any() else 1e-8

        self.propensity_table = self.propensity_table / self.propensity_table.sum(axis=1, keepdims=True)

    def predict_proba(self, X: np.ndarray, M: int) -> np.ndarray:
        x_flat = x_to_flat(X, self.feature_sizes)
        return self.propensity_table[x_flat]


class LearnedPropensityModel:
    """Learned propensity model p(t | x) using linear or MLP."""

    def __init__(
        self,
        model_type: Literal["linear", "mlp"] = "mlp",
        lr: float = 0.01,
        epochs: int = 5000,
        batch_size: int = 256,
        use_minibatch: bool = False,
        verbose: bool = False,
        balance_reg: float = 0.0,
        norm_reg: float = 0.0,
        balance_moments: int = 1,
        **model_kwargs,
    ):
        """Initialize learned propensity model.

        Args:
            model_type: "linear" or "mlp"
            lr: Learning rate
            epochs: Number of training epochs
            batch_size: Batch size for training
            use_minibatch: If True, use minibatch SGD; if False, use full batch GD
            verbose: Whether to print training progress
            balance_reg: Weight on balance regularization
            norm_reg: Weight on normalization regularization
            balance_moments: Number of standardized moments to include in balance loss
            model_kwargs: Architecture-specific hyperparameters
        """
        self.balance_reg = balance_reg
        self.norm_reg = norm_reg
        self.balance_moments = balance_moments

        shared = dict(lr=lr, epochs=epochs, batch_size=batch_size,
                      use_minibatch=use_minibatch, verbose=verbose)

        if model_type == "linear":
            self.model = LinearModel(task="classification", **shared)
        elif model_type == "mlp":
            self.model = MLPModel(task="classification", **shared, **model_kwargs)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Must be 'linear' or 'mlp'.")

    def fit(self, X: np.ndarray, T: np.ndarray, M: int,
            val_X: np.ndarray, val_T: np.ndarray, **kwargs) -> None:
        """Train propensity model.

        Args:
            X: (n, p) confounders
            T: (n,) treatment assignments
            M: Number of treatments
            val_X: (m, p) validation confounders
            val_T: (m,) validation treatments
            kwargs: Additional keyword arguments, e.g. true_E_X: (p,)
        """
        if self.balance_reg > 0.0 or self.norm_reg > 0.0:
            balance_moment_targets = kwargs['balance_moment_targets']
            moment_targets_t = {k: torch.tensor(v, dtype=torch.float32)
                                 for k, v in balance_moment_targets.items()}
            criterion = BalNormCriterion(self.balance_reg, self.norm_reg,
                                         balance_moments=self.balance_moments,
                                         moment_targets=moment_targets_t)
        else:
            criterion = None
        self.model.fit(
            X, T,
            input_dim=X.shape[1],
            output_dim=M,
            criterion=criterion,
            pass_inputs_to_criterion=(self.balance_reg > 0.0 or self.norm_reg > 0.0),
            val_X=val_X,
            val_y=val_T,
        )

    def predict_proba(self, X: np.ndarray, M: int) -> np.ndarray:
        return self.model.predict(X)

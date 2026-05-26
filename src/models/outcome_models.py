"""
Outcome models for predicting Y from (X, T).

Used in outcome imputation methods.
All models follow a common interface:
- fit(X, T, Y, M, **kwargs): Train the model
- predict(X, T, M): Predict outcomes for given (X, T) pairs
"""

import numpy as np
from typing import Literal

from .linear import LinearModel
from .mlp import MLPModel


class OutcomeModel:
    """
    Outcome model μ(x, t) using linear or MLP.

    Predicts outcomes from confounders and treatment assignments.
    """

    def __init__(
        self,
        model_type: Literal["linear", "mlp"] = "mlp",
        hidden_dim: int = 8,
        lr: float = 0.01,
        epochs: int = 3000,
        batch_size: int = 256,
        use_minibatch: bool = False,
        verbose: bool = False
    ):
        """Initialize outcome model.

        Args:
            model_type: "linear" or "mlp"
            hidden_dim: Hidden dimension (only used for MLP)
            lr: Learning rate
            epochs: Number of training epochs
            batch_size: Batch size for training
            use_minibatch: If True, use minibatch SGD; if False, use full batch GD
            verbose: Whether to print training progress
        """
        self.model_type = model_type

        if model_type == "linear":
            self.model = LinearModel(
                task="regression",
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                use_minibatch=use_minibatch,
                verbose=verbose
            )
        elif model_type == "mlp":
            self.model = MLPModel(
                hidden_dim=hidden_dim,
                task="regression",
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                use_minibatch=use_minibatch,
                verbose=verbose
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Must be 'linear' or 'mlp'.")

    def fit(self, X: np.ndarray, T_multi: np.ndarray, Y: np.ndarray, M: int,
            val_X: np.ndarray, val_T_multi: np.ndarray, val_Y: np.ndarray,
            **kwargs) -> None:
        """Train outcome model.

        Args:
            X: (n, d_x) confounders
            T_multi: (n, d_t) treatment feature vectors
            Y: (n,) observed outcomes
            M: Number of treatments
            val_X: (m, d_x) validation confounders for train/val comparison
            val_T_multi: (m, d_t) validation treatment features
            val_Y: (m,) validation outcomes
        """
        input_dim = X.shape[1] + T_multi.shape[1]
        X_concat = np.concatenate([X, T_multi], axis=1)
        val_X_concat = np.concatenate([val_X, val_T_multi], axis=1)
        self.model.fit(X_concat, Y, input_dim=input_dim, output_dim=1,
                       val_X=val_X_concat, val_y=val_Y)

    def predict(self, X: np.ndarray, T_multi: np.ndarray, M: int) -> np.ndarray:
        X_concat = np.concatenate([X, T_multi], axis=1)
        return self.model.predict(X_concat)

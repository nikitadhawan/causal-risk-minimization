"""
APO (Average Potential Outcome) model g(t).

All models follow a common interface:
- fit(T, Y, M, **kwargs): Train the model
- predict(t_values): Return APO estimates for given treatments
"""

import numpy as np
from typing import Literal

from .linear import LinearModel
from .mlp import MLPModel


class APOModel:
    """
    APO model g(t) using linear or MLP.

    Predicts average potential outcomes from treatment assignments.
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
        """Initialize APO model.

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

        self.M: int = None

    def fit(self, T_multi: np.ndarray, Y: np.ndarray, M: int,
            val_T_multi: np.ndarray, val_Y: np.ndarray,
            weights: np.ndarray = None,
            val_weights: np.ndarray = None,
            **kwargs) -> None:
        """Train APO model.

        Args:
            T_multi:  (n, d_t) treatment feature vectors
            Y:        (n,) observed outcomes
            M:        Number of treatments
            val_T_multi: (m, d_t) validation treatment features for train/val comparison
            val_Y:       (m,) validation outcomes
            weights:     (n,) importance weights. If provided, trains on Y * weights.
            val_weights: (m,) validation importance weights. If provided, val targets = val_Y * val_weights.
        """
        self.M = M
        targets = Y if weights is None else Y * weights
        val_targets = val_Y if val_weights is None else val_Y * val_weights
        self.model.fit(T_multi, targets, input_dim=T_multi.shape[1], output_dim=1,
                       val_X=val_T_multi, val_y=val_targets)

    def predict(self, T_multi: np.ndarray, M: int) -> np.ndarray:
        """Predict APOs for given treatment feature vectors.

        Args:
            T_multi: (m, d_t) treatment feature vectors

        Returns:
            (m,) APO estimates
        """
        return self.model.predict(T_multi)

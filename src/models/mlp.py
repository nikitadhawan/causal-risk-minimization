"""Generic Multi-Layer Perceptron (MLP) for classification or regression."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Literal

from ..utils import train_model_batch, train_model_minibatch, get_device


class MLPModel(nn.Module):
    """
    Generic trainable 2-layer MLP with ReLU activations.
    Linear -> ReLU -> Linear -> ReLU -> Linear

    Can be used for:
    - Propensity models (classification)
    - APO models (regression)
    - Outcome models (regression)
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        task: Literal["classification", "regression"] = "regression",
        lr: float = 0.01,
        epochs: int = 1000,
        batch_size: int = 256,
        use_minibatch: bool = False,
        verbose: bool = False
    ):
        """
        Args:
            hidden_dim: Hidden layer dimension
            task: "classification" or "regression"
            lr: Learning rate
            epochs: Number of training epochs
            batch_size: Batch size for training (only used if use_minibatch=True)
            use_minibatch: If True, use minibatch SGD; if False, use full batch GD
            verbose: Whether to print training progress
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.task = task
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_minibatch = use_minibatch
        self.verbose = verbose
        self.device = get_device()
        self.network: nn.Sequential = None
        self._is_fitted = False        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.network(x)
        # For regression with output_dim=1, squeeze the last dimension
        if output.shape[-1] == 1:
            return output.squeeze(-1)
        return output

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        input_dim: int,
        output_dim: int,
        val_X: np.ndarray,
        val_y: np.ndarray,
        criterion: nn.Module = None,
        pass_inputs_to_criterion: bool = False
    ) -> None:
        """
        Train the MLP model.

        Args:
            X: (n, input_dim) input features
            y: (n,) targets (class labels for classification, values for regression)
            input_dim: Input dimension
            output_dim: Output dimension (num classes for classification, 1 for regression)
            val_X: (m, input_dim) validation inputs for train/val loss comparison
            val_y: (m,) validation targets for train/val loss comparison
            criterion: Loss function
            pass_inputs_to_criterion: If True, call criterion(predictions, target, *inputs)
        """
        self.network = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, output_dim)
        )
        self.to(self.device)

        X_tensor = torch.FloatTensor(X)

        if self.task == "classification":
            y_tensor = torch.LongTensor(y)
            base_criterion = nn.CrossEntropyLoss()
        else:  # regression
            y_tensor = torch.FloatTensor(y)
            base_criterion = nn.MSELoss()

        if criterion is None:
            criterion = base_criterion
        else:
            assert hasattr(criterion, 'base_criterion')
            criterion.base_criterion = base_criterion

        val_inputs_t = (torch.FloatTensor(val_X),)
        val_target_t = (torch.LongTensor(val_y) if self.task == "classification"
                        else torch.FloatTensor(val_y))

        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        if self.use_minibatch:
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(
                TensorDataset(*val_inputs_t, val_target_t),
                batch_size=self.batch_size, shuffle=False,
            )

            train_model_minibatch(
                model=self,
                dataloader=dataloader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                epochs=self.epochs,
                verbose=self.verbose,
                model_name="MLP",
                device=self.device,
                pass_inputs_to_criterion=pass_inputs_to_criterion
            )
        else:
            # Full batch training
            train_model_batch(
                model=self,
                inputs=(X_tensor,),
                target=y_tensor,
                val_inputs=val_inputs_t,
                val_target=val_target_t,
                optimizer=optimizer,
                criterion=criterion,
                epochs=self.epochs,
                verbose=self.verbose,
                model_name="MLP",
                device=self.device,
                pass_inputs_to_criterion=pass_inputs_to_criterion
            )

        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using trained model.

        Args:
            X: (n, input_dim) input features

        Returns:
            For classification: (n, output_dim) class probabilities
            For regression: (n,) predicted values
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_tensor = torch.FloatTensor(X).to(self.device)
        self.eval()

        with torch.no_grad():
            output = self(X_tensor)

            if self.task == "classification":
                probs = torch.softmax(output, dim=1)
                return probs.cpu().numpy()
            else:  # regression
                return output.cpu().numpy()

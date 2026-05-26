"""Utility functions."""

import random
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def compute_x_moments(X: np.ndarray, k: int) -> Dict:
    """Compute empirical standardized moments of X up to order k.

    X is z-scored (subtract mean, divide by std) before computing moments so
    that all targets are O(1) regardless of order j, preventing higher-order
    terms from dominating the balance loss.

    Args:
        X: (n, d_x) covariate matrix
        k: Maximum moment order

    Returns:
        Dict with:
          j (int)   -> (d_x,) array of E[Z^j] for j = 1..k, where Z = (X-μ)/σ
          'mean'    -> (d_x,) per-feature mean μ
          'std'     -> (d_x,) per-feature std σ (epsilon-floored)
    """
    X = X.astype(np.float32)
    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8
    Z    = (X - mean) / std
    return {j: (Z ** j).mean(axis=0) for j in range(1, k + 1)} | {'mean': mean, 'std': std}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model_minibatch(
    model: nn.Module,
    dataloader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    verbose: bool = False,
    model_name: str = "Model",
    device: torch.device = None,
    pass_inputs_to_criterion: bool = False
) -> None:
    """
    Mini-batch training loop.

    Args:
        model: PyTorch model to train
        dataloader: DataLoader providing batches of (*inputs, target)
        val_loader: DataLoader for validation data
        optimizer: Optimizer for training
        criterion: Loss function
        epochs: Number of training epochs
        verbose: Whether to print progress roughly 10 times
        model_name: Name for verbose printing
        device: Device to train on (auto-detected if None)
        pass_inputs_to_criterion: If True, pass auxiliary inputs to criterion (e.g. balance)
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    print_freq = max(1, epochs // 10)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            *inputs, target = batch
            inputs = [inp.to(device) for inp in inputs]
            target = target.to(device)

            optimizer.zero_grad()
            predictions = model(*inputs)
            if pass_inputs_to_criterion:
                loss = criterion(predictions, target, *inputs)
            else:
                loss = criterion(predictions, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if verbose and (epoch + 1) % print_freq == 0:
            avg_loss = total_loss / len(dataloader)
            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for val_batch in val_loader:
                    *val_inputs, val_target = val_batch
                    val_inputs = [inp.to(device) for inp in val_inputs]
                    val_target = val_target.to(device)
                    val_preds = model(*val_inputs)
                    if pass_inputs_to_criterion:
                        val_total += criterion(val_preds, val_target, *val_inputs).item()
                    else:
                        val_total += criterion(val_preds, val_target).item()
            val_loss = val_total / len(val_loader)
            print(f"  {model_name} epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f} (val: {val_loss:.4f})")


def train_model_batch(
    model: nn.Module,
    inputs: tuple,
    target: torch.Tensor,
    val_inputs: tuple,
    val_target: torch.Tensor,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    verbose: bool = False,
    model_name: str = "Model",
    device: torch.device = None,
    pass_inputs_to_criterion: bool = False
) -> None:
    """
    Full-batch training loop.

    Args:
        model: PyTorch model to train
        inputs: Tuple of input tensors (*inputs,)
        target: Target tensor
        val_inputs: Tuple of validation input tensors
        val_target: Validation target tensor
        optimizer: Optimizer for training
        criterion: Loss function
        epochs: Number of training epochs
        verbose: Whether to print progress roughly 10 times
        model_name: Name for verbose printing
        device: Device to train on (auto-detected if None)
        pass_inputs_to_criterion: If True, pass auxiliary inputs to criterion (e.g. balance)
    """
    if device is None:
        device = get_device()

    model = model.to(device)

    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    inputs = tuple(inp.to(device) for inp in inputs)
    target = target.to(device)

    print_freq = max(1, epochs // 10)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(*inputs)
        if pass_inputs_to_criterion:
            loss = criterion(predictions, target, *inputs)
        else:
            loss = criterion(predictions, target)
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % print_freq == 0:
            train_loss = loss.item()
            model.eval()
            val_inputs = tuple(inp.to(device) for inp in val_inputs)
            val_target = val_target.to(device)
            with torch.no_grad():
                val_preds = model(*val_inputs)
                if pass_inputs_to_criterion:
                    val_loss = criterion(val_preds, val_target, *val_inputs).item()
                else:
                    val_loss = criterion(val_preds, val_target).item()
            print(f"  {model_name} epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f} (val: {val_loss:.4f})")


def x_to_flat(X: np.ndarray, feature_sizes: list) -> np.ndarray:
    """Convert (n, d_x) discrete features to (n,) flat row indices (row-major).

    For 1-D X (d_x=1) this is just X[:, 0].  For multi-dim X the index is
    computed in the same row-major order used by SyntheticTextDataset._flat_index.
    """
    if X.shape[1] == 1:
        return X[:, 0].astype(int)
    idx = np.zeros(len(X), dtype=int)
    stride = 1
    for j in range(len(feature_sizes) - 1, -1, -1):
        idx += X[:, j].astype(int) * stride
        stride *= feature_sizes[j]
    return idx


def tokenize_xt(
    X_scalar: np.ndarray, T: np.ndarray, n_x_values: int, M: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Assign integer tokens to X and T values for transformer input.

    X values {0, ..., n_x_values-1} map to tokens {0, ..., n_x_values-1}.
    T values {0, ..., M-1} map to tokens {n_x_values, ..., n_x_values+M-1}.

    Returns:
        x_tokens: (n,) integer tokens for X
        t_tokens: (n,) integer tokens for T
        vocab_size: n_x_values + M
    """
    return X_scalar.copy(), T + n_x_values, n_x_values + M
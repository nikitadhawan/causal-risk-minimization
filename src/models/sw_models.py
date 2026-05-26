"""
Stabilized weight models for directly estimating w(x,t) = p(t)/p(t|x).

All models follow a common interface:
- fit(X, T_multi, all_T_multi, T, M): Train the model
- predict_weights(X, T_multi): Return (n,) array of stabilized weights
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Literal

from ..utils import get_device


class SWModel:
    """
    Stabilized weight model (linear or MLP).

    Directly estimates w(x,t) = p(t)/p(t|x) as a positive scalar using a
    linear layer (or MLP) followed by softplus activation.

    Trained with a balance + normalization loss equivalent to BalNormCriterion,
    expressed directly in terms of w rather than propensity logits:

      balance_loss = sum_t sum_k || E_w[X^k | T=t] - E[X^k] ||^2
      norm_loss = sum_t (E[w_i | T=t] - 1)^2

    Input features are the concatenation of X and T_multi.
    """

    def __init__(
        self,
        model_type: Literal["linear", "mlp"] = "linear",
        hidden_dim: int = 64,
        balance_weight: float = 1.0,
        norm_weight: float = 1.0,
        balance_moments: int = 1,
        lr: float = 0.01,
        epochs: int = 500,
        batch_size: int = 256,
        use_minibatch: bool = False,
        verbose: bool = False,
    ):
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.balance_weight = balance_weight
        self.norm_weight = norm_weight
        self.balance_moments = balance_moments
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_minibatch = use_minibatch
        self.verbose = verbose
        self.device = get_device()
        self._net = None

    def _build_net(self, input_dim: int) -> nn.Sequential:
        if self.model_type == "linear":
            return nn.Sequential(
                nn.Linear(input_dim, 1),
                nn.Softplus(),
            )
        else:  # mlp
            return nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
                nn.Softplus(),
            )

    def _compute_loss(
        self,
        w: torch.Tensor,
        X_t: torch.Tensor,
        T_t: torch.Tensor,
        M: int,
        moment_targets: Dict,
    ) -> torch.Tensor:
        balance_loss = torch.tensor(0.0, device=w.device)
        norm_loss = torch.tensor(0.0, device=w.device)

        mean = moment_targets['mean'].to(w.device)
        std  = moment_targets['std'].to(w.device)
        Z_t  = (X_t - mean) / std

        for t in range(M):
            mask = T_t == t
            if mask.sum() == 0:
                continue

            w_t = w[mask]  # (n_t,)

            if self.balance_weight > 0.0:
                for j in range(1, self.balance_moments + 1):
                    Z_pow = Z_t[mask] ** j
                    weighted_mean = (w_t.unsqueeze(1) * Z_pow).mean(dim=0) 
                    balance_loss = balance_loss + ((weighted_mean - moment_targets[j].to(w.device)) ** 2).mean() / (2 ** j)

            if self.norm_weight > 0.0:
                norm_loss = norm_loss + (w_t.mean() - 1.0) ** 2

        return self.balance_weight * balance_loss + self.norm_weight * norm_loss

    def fit(
        self,
        X: np.ndarray,
        T_multi: np.ndarray,
        T: np.ndarray,
        M: int,
        val_X: np.ndarray,
        val_T_multi: np.ndarray,
        val_T: np.ndarray,
        balance_moment_targets: Dict[int, np.ndarray],
        **kwargs,
    ) -> None:
        """
        Train the stabilized weight model.

        Args:
            X: (n, d_x) confounders
            T_multi: (n, d_t) observed treatment feature vectors
            T: (n,) flat treatment indices
            M: Number of treatments
            val_X: (m, d_x) validation confounders for train/val comparison
            val_T_multi: (m, d_t) validation treatment feature vectors
            val_T: (m,) validation flat treatment indices
            balance_moment_targets: Dict mapping j -> (d_x,) array of E[X^j] for j=1..k, precomputed from training data.
        """
        XT = np.concatenate([X, T_multi], axis=1).astype(np.float32)  # (n, d_x+d_t)

        input_dim = XT.shape[1]
        self._net = self._build_net(input_dim).to(self.device)
        optimizer = optim.Adam(self._net.parameters(), lr=self.lr)

        XT_t = torch.from_numpy(XT).to(self.device)
        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
        T_t = torch.from_numpy(T).long().to(self.device)
        moment_targets_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in balance_moment_targets.items()}
        val_XT = np.concatenate([val_X, val_T_multi], axis=1).astype(np.float32)
        val_XT_t = torch.from_numpy(val_XT)
        val_X_t = torch.from_numpy(val_X.astype(np.float32))
        val_T_t = torch.from_numpy(val_T).long()

        model_name = f"SWModel({self.model_type})"
        print_freq = max(1, self.epochs // 10)
        if self.use_minibatch:
            dataset = TensorDataset(XT_t, X_t, T_t)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(val_XT_t, val_X_t, val_T_t), batch_size=self.batch_size, shuffle=False)
            for epoch in range(self.epochs):
                self._net.train()
                epoch_loss = 0.0
                for XT_batch, X_batch, T_batch in dataloader:
                    optimizer.zero_grad()
                    w_batch = self._net(XT_batch).squeeze(-1)
                    loss = self._compute_loss(w_batch, X_batch, T_batch, M, moment_targets=moment_targets_t)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                if self.verbose and (epoch + 1) % print_freq == 0:
                    self._net.eval()
                    val_total = 0.0
                    with torch.no_grad():
                        for vXT, vX, vT in val_loader:
                            vXT, vX, vT = vXT.to(self.device), vX.to(self.device), vT.to(self.device)
                            vw = self._net(vXT).squeeze(-1)
                            val_total += self._compute_loss(vw, vX, vT, M, moment_targets=moment_targets_t).item()
                    print(f"  [{model_name}] epoch {epoch+1}/{self.epochs}, loss={epoch_loss:.4f} (val: {val_total/len(val_loader):.4f})")
        else:
            for epoch in range(self.epochs):
                self._net.train()
                optimizer.zero_grad()
                w = self._net(XT_t).squeeze(-1)  # (n,)
                loss = self._compute_loss(w, X_t, T_t, M, moment_targets=moment_targets_t)
                loss.backward()
                optimizer.step()
                if self.verbose and (epoch + 1) % print_freq == 0:
                    self._net.eval()
                    with torch.no_grad():
                        val_XT_t = val_XT_t.to(self.device)
                        val_X_t = val_X_t.to(self.device)
                        val_T_t = val_T_t.to(self.device)
                        val_w = self._net(val_XT_t).squeeze(-1)
                        val_loss = self._compute_loss(val_w, val_X_t, val_T_t, M, moment_targets=moment_targets_t).item()
                    print(f"  [{model_name}] epoch {epoch+1}/{self.epochs}, loss={loss.item():.4f} (val: {val_loss:.4f})")

    def predict_weights(self, X: np.ndarray, T_multi: np.ndarray) -> np.ndarray:
        """
        Predict stabilized weights w(x,t).

        Args:
            X: (n, d_x) confounders
            T_multi: (n, d_t) treatment feature vectors

        Returns:
            (n,) stabilized weights
        """
        self._net.eval()
        XT = np.concatenate([X, T_multi], axis=1).astype(np.float32)
        XT_t = torch.from_numpy(XT).to(self.device)
        with torch.no_grad():
            w = self._net(XT_t).squeeze(-1).cpu().numpy()
        return w

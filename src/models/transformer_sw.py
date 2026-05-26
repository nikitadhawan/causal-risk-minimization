"""Transformer stabilized weight model: w(x, t) via bidirectional encoding + pooling."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict

from .transformer import TransformerBlock
from ..utils import get_device


class TransformerSWModel(nn.Module):
    """
    Transformer stabilized weight model.

    Encodes the full [x_0, ..., x_{d_x-1}, t_0, ..., t_{d_t-1}] sequence with a
    causal (unidirectional) transformer, takes the last token's hidden state, then
    projects to a scalar w(x,t) = p(t)/p(t|x) via a linear head + softplus.

    Vocabulary layout (matches TransformerPropensityModel):
      X features:    [0,           sum(feature_sizes))
      T dimension j: [sum_feat + sum(treatment_sizes[:j]),
                      sum_feat + sum(treatment_sizes[:j+1]))

    Trained with the same balance + normalization loss as SWModel:
      balance_loss = sum_t || E_w[X | T=t] - E[X] ||^2
      norm_loss    = sum_t (E[w_i | T=t] - 1)^2

    Architecture is built lazily in fit() once feature_sizes and treatment_sizes
    are known, so the model can be instantiated from config before the dataset
    is loaded.
    """

    def __init__(
        self,
        d_model: int = 8,
        num_heads: int = 2,
        d_hidden: int = 32,
        num_blocks: int = 2,
        balance_weight: float = 1.0,
        norm_weight: float = 1.0,
        balance_moments: int = 1,
        lr: float = 0.0001,
        epochs: int = 2000,
        batch_size: int = 256,
        use_minibatch: bool = False,
        verbose: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_hidden = d_hidden
        self.num_blocks = num_blocks
        self.balance_weight = balance_weight
        self.norm_weight = norm_weight
        self.balance_moments = balance_moments
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_minibatch = use_minibatch
        self.verbose = verbose
        self.device = get_device()
        self._is_fitted = False
        # Set in fit()
        self._x_offsets: np.ndarray = None
        self._sum_feat: int = None
        self._t_offsets: np.ndarray = None
        # Layers built in fit()
        self.embedding: nn.Embedding = None
        self.blocks: nn.ModuleList = None
        self.head: nn.Sequential = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) LongTensor of token indices
        Returns:
            (batch,) stabilized weights (positive via softplus)
        """
        emb = self.embedding(x)       # (batch, seq_len, d_model)
        for block in self.blocks:
            emb = block(emb)
        return self.head(emb[:, -1, :]).squeeze(-1)  # last token → (batch,)

    def _build(self, vocab_size: int) -> None:
        self.embedding = nn.Embedding(vocab_size, self.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(self.d_model, self.num_heads, self.d_hidden, causal=True)
            for _ in range(self.num_blocks)
        ])
        self.head = nn.Sequential(
            nn.Linear(self.d_model, 1),
            nn.Softplus(),
        )
        self.to(self.device)

    def _tokens(self, X: np.ndarray, T_multi: np.ndarray) -> torch.Tensor:
        """Encode [X, T_multi] into token indices. Returns (n, d_x+d_t) LongTensor."""
        X_tokens = X.astype(int) + self._x_offsets[np.newaxis, :]
        T_tokens = T_multi.astype(int) + self._sum_feat + self._t_offsets[np.newaxis, :]
        return torch.LongTensor(np.concatenate([X_tokens, T_tokens], axis=1))

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

            w_t = w[mask]

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
        balance_moment_targets: Dict,
        feature_sizes: list = None,
        treatment_sizes: list = None,
        **kwargs,
    ) -> None:
        """
        Train the transformer stabilized weight model.

        Args:
            X: (n, d_x) integer-valued confounders
            T_multi: (n, d_t) per-dimension treatment values
            T: (n,) flat treatment indices
            M: Number of treatments
            val_X: (m, d_x) validation confounders
            val_T_multi: (m, d_t) validation treatment values
            val_T: (m,) validation flat treatment indices
            feature_sizes: Cardinalities per X dimension
            treatment_sizes: Cardinalities per T dimension. Defaults to [M].
            balance_moment_targets: Dict mapping j -> (d_x,) array of E[X^j] for j=1..k,
                precomputed from training data.
        """
        if feature_sizes is None:
            raise ValueError("feature_sizes must be provided")
        if treatment_sizes is None:
            treatment_sizes = [M]

        self._x_offsets = np.concatenate([[0], np.cumsum(feature_sizes[:-1])])
        self._sum_feat = sum(feature_sizes)
        self._t_offsets = np.concatenate([[0], np.cumsum(treatment_sizes[:-1])])

        vocab_size = self._sum_feat + sum(treatment_sizes)
        self._build(vocab_size)

        seq_tensor = self._tokens(X, T_multi)
        X_t = torch.FloatTensor(X)
        T_t = torch.LongTensor(T.astype(int))
        moment_targets_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in balance_moment_targets.items()}
        val_seq = self._tokens(val_X, val_T_multi)
        val_Xf = torch.FloatTensor(val_X)
        val_Tf = torch.LongTensor(val_T.astype(int))

        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        print_freq = max(1, self.epochs // 10)

        if self.use_minibatch:
            dataset = TensorDataset(seq_tensor, X_t, T_t)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(val_seq, val_Xf, val_Tf), batch_size=self.batch_size, shuffle=False)
            for epoch in range(self.epochs):
                self.train()
                total_loss = 0.0
                for seq_b, X_b, T_b in dataloader:
                    seq_b = seq_b.to(self.device)
                    X_b = X_b.to(self.device)
                    T_b = T_b.to(self.device)
                    optimizer.zero_grad()
                    w = self(seq_b)
                    loss = self._compute_loss(w, X_b, T_b, M, moment_targets=moment_targets_t)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                if self.verbose and (epoch + 1) % print_freq == 0:
                    avg_loss = total_loss / len(dataloader)
                    self.eval()
                    val_total = 0.0
                    with torch.no_grad():
                        for vs, vx, vt in val_loader:
                            vs, vx, vt = vs.to(self.device), vx.to(self.device), vt.to(self.device)
                            vw = self(vs)
                            val_total += self._compute_loss(vw, vx, vt, M, moment_targets=moment_targets_t).item()
                    print(f"  TransformerSW epoch {epoch+1}/{self.epochs}, "
                          f"loss={avg_loss:.4f} (val: {val_total / len(val_loader):.4f})")
        else:
            seq_tensor = seq_tensor.to(self.device)
            X_t = X_t.to(self.device)
            T_t = T_t.to(self.device)
            for epoch in range(self.epochs):
                self.train()
                optimizer.zero_grad()
                w = self(seq_tensor)
                loss = self._compute_loss(w, X_t, T_t, M, moment_targets=moment_targets_t)
                loss.backward()
                optimizer.step()
                if self.verbose and (epoch + 1) % print_freq == 0:
                    self.eval()
                    val_seq_t = val_seq.to(self.device)
                    val_X_t = val_Xf.to(self.device)
                    val_T_t = val_Tf.to(self.device)
                    with torch.no_grad():
                        val_w = self(val_seq_t)
                        val_loss = self._compute_loss(val_w, val_X_t, val_T_t, M, moment_targets=moment_targets_t).item()
                    print(f"  TransformerSW epoch {epoch+1}/{self.epochs}, "
                          f"loss={loss.item():.4f} (val: {val_loss:.4f})")

        self._is_fitted = True

    def predict_weights(self, X: np.ndarray, T_multi: np.ndarray) -> np.ndarray:
        """
        Predict stabilized weights w(x,t).

        Args:
            X: (n, d_x) confounders
            T_multi: (n, d_t) treatment feature vectors

        Returns:
            (n,) stabilized weights
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        seq_tensor = self._tokens(X, T_multi).to(self.device)
        self.eval()
        with torch.no_grad():
            w = self(seq_tensor).cpu().numpy()
        return w

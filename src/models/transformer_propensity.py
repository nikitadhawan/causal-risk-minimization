"""Transformer propensity model: p(t | x) via autoregressive factorization."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Optional

from .transformer import TransformerBlock
from .propensity_models import BalNormCriterion
from ..utils import get_device


class TransformerPropensityModel(nn.Module):
    """
    Autoregressive transformer propensity model p(t | x).

    Vocabulary layout (non-overlapping tokens):
      X features:    [0,           sum(feature_sizes))
      T dimension j: [sum_feat + sum(treatment_sizes[:j]),
                      sum_feat + sum(treatment_sizes[:j+1]))

    Training (teacher-forced):
      Input:  [x_0, ..., x_{d_x-1}, t_0, ..., t_{d_t-2}]
      Target: t_0, ..., t_{d_t-1} predicted at positions d_x-1, d_x, ..., d_x+d_t-2

    Inference (autoregressive):
      d_t forward passes building up the sequence token by token.
      For d_t == 1 a single forward pass suffices (fast path).

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
        lr: float = 0.0001,
        epochs: int = 2000,
        batch_size: int = 256,
        use_minibatch: bool = False,
        verbose: bool = False,
        balance_reg: float = 0.0,
        norm_reg: float = 0.0,
        balance_moments: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_hidden = d_hidden
        self.num_blocks = num_blocks
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_minibatch = use_minibatch
        self.verbose = verbose
        self.balance_reg = balance_reg
        self.norm_reg = norm_reg
        self.balance_moments = balance_moments
        self.device = get_device()
        self._is_fitted = False
        # Set in fit()
        self._M: int = None
        self._d_x: int = None
        self._sum_feat: int = None
        self._x_offsets: np.ndarray = None
        self._treatment_sizes: list = None
        self._t_offsets: np.ndarray = None
        self._sum_t: int = None
        self._criterion: nn.Module = None
        self._all_t_multi_tensor: torch.Tensor = None
        # Layers built in fit()
        self.embedding: nn.Embedding = None
        self.blocks: nn.ModuleList = None
        self.unembed: nn.Linear = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len) LongTensor. Returns (batch, seq_len, vocab_size) logits."""
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.unembed(x)

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        M: int,
        val_X: np.ndarray,
        val_T: np.ndarray,
        val_T_multi: np.ndarray,
        feature_sizes: list = None,
        treatment_sizes: list = None,
        T_multi: np.ndarray = None,
        balance_moment_targets: Dict = None,
        **kwargs,
    ) -> None:
        """
        Args:
            X:                      (n, d_x) integer-valued confounders
            T:                      (n,) flat treatment indices
            M:                      Number of treatments
            val_X:                  (m, d_x) validation confounders
            val_T:                  (m,) validation flat treatment indices
            val_T_multi:            (m, d_t) validation treatment values
            feature_sizes:          Cardinalities per X dimension
            treatment_sizes:        Cardinalities per T dimension. Defaults to [M].
            T_multi:                (n, d_t) per-dimension treatment values
            balance_moment_targets: {j: E[Z^j], 'mean': μ, 'std': σ} for balance reg
        """
        if feature_sizes is None:
            raise ValueError("feature_sizes must be provided")
        if T_multi is None:
            raise ValueError("T_multi must be provided")

        if treatment_sizes is None:
            treatment_sizes = [M]

        self._M = M
        self._d_x = X.shape[1]
        self._x_offsets = np.concatenate([[0], np.cumsum(feature_sizes[:-1])])
        self._sum_feat = sum(feature_sizes)
        self._treatment_sizes = treatment_sizes
        self._t_offsets = np.concatenate([[0], np.cumsum(treatment_sizes[:-1])])
        self._sum_t = sum(treatment_sizes)

        vocab_size = self._sum_feat + self._sum_t
        self.embedding = nn.Embedding(vocab_size, self.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(self.d_model, self.num_heads, self.d_hidden)
            for _ in range(self.num_blocks)
        ])
        self.unembed = nn.Linear(self.d_model, vocab_size)
        self.to(self.device)

        grids = np.meshgrid(*[np.arange(s) for s in treatment_sizes], indexing='ij')
        self._all_t_multi_tensor = torch.LongTensor(
            np.column_stack([g.ravel() for g in grids])
        ).to(self.device)

        if self.balance_reg > 0.0 or self.norm_reg > 0.0:
            moment_targets_t = {k: torch.tensor(v, dtype=torch.float32)
                                 for k, v in balance_moment_targets.items()}
            self._criterion = BalNormCriterion(self.balance_reg, self.norm_reg,
                                               balance_moments=self.balance_moments,
                                               moment_targets=moment_targets_t)
        else:
            self._criterion = None

        X_tokens = X.astype(int) + self._x_offsets[np.newaxis, :]
        T_tokens = T_multi.astype(int) + self._sum_feat + self._t_offsets[np.newaxis, :]

        X_tensor = torch.LongTensor(X_tokens)
        T_tok_tensor = torch.LongTensor(T_tokens)
        d_t = len(treatment_sizes)
        if d_t > 1:
            seq_tensor = torch.cat([X_tensor, T_tok_tensor[:, :-1]], dim=1)
        else:
            seq_tensor = X_tensor

        X_float_tensor = torch.FloatTensor(X)
        T_flat_tensor = torch.LongTensor(T.astype(int))

        val_X_tokens = val_X.astype(int) + self._x_offsets[np.newaxis, :]
        val_T_tokens = val_T_multi.astype(int) + self._sum_feat + self._t_offsets[np.newaxis, :]
        val_X_tok = torch.LongTensor(val_X_tokens)
        val_T_tok = torch.LongTensor(val_T_tokens)
        if d_t > 1:
            val_seq = torch.cat([val_X_tok, val_T_tok[:, :-1]], dim=1)
        else:
            val_seq = val_X_tok
        val_X_float = torch.FloatTensor(val_X)
        val_T_flat = torch.LongTensor(val_T.astype(int))

        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        print_freq = max(1, self.epochs // 10)

        if self.use_minibatch:
            dataset = TensorDataset(seq_tensor, T_tok_tensor, X_float_tensor, T_flat_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(
                TensorDataset(val_seq, val_T_tok, val_X_float, val_T_flat),
                batch_size=self.batch_size, shuffle=False,
            )
            for epoch in range(self.epochs):
                self.train()
                total_loss = 0.0
                for seq_b, t_tok_b, x_b, t_flat_b in dataloader:
                    seq_b = seq_b.to(self.device)
                    t_tok_b = t_tok_b.to(self.device)
                    x_b = x_b.to(self.device)
                    t_flat_b = t_flat_b.to(self.device)
                    optimizer.zero_grad()
                    loss = self._compute_loss(seq_b, t_tok_b, x_b, t_flat_b)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                if self.verbose and (epoch + 1) % print_freq == 0:
                    avg_loss = total_loss / len(dataloader)
                    self.eval()
                    val_total = 0.0
                    with torch.no_grad():
                        for vs, vt, vx, vtf in val_loader:
                            vs, vt, vx, vtf = (vs.to(self.device), vt.to(self.device),
                                               vx.to(self.device), vtf.to(self.device))
                            val_total += self._compute_loss(vs, vt, vx, vtf).item()
                    print(f"  TransformerPropensity epoch {epoch+1}/{self.epochs}, "
                          f"Loss: {avg_loss:.4f} (val: {val_total / len(val_loader):.4f})")
        else:
            seq_tensor = seq_tensor.to(self.device)
            T_tok_tensor = T_tok_tensor.to(self.device)
            X_float_tensor = X_float_tensor.to(self.device)
            T_flat_tensor = T_flat_tensor.to(self.device)
            for epoch in range(self.epochs):
                self.train()
                optimizer.zero_grad()
                loss = self._compute_loss(seq_tensor, T_tok_tensor, X_float_tensor, T_flat_tensor)
                loss.backward()
                optimizer.step()
                if self.verbose and (epoch + 1) % print_freq == 0:
                    self.eval()
                    val_seq_t = val_seq.to(self.device)
                    val_T_tok_t = val_T_tok.to(self.device)
                    val_X_float_t = val_X_float.to(self.device)
                    val_T_flat_t = val_T_flat.to(self.device)
                    with torch.no_grad():
                        val_loss = self._compute_loss(
                            val_seq_t, val_T_tok_t, val_X_float_t, val_T_flat_t
                        ).item()
                    print(f"  TransformerPropensity epoch {epoch+1}/{self.epochs}, "
                          f"Loss: {loss.item():.4f} (val: {val_loss:.4f})")

        self._is_fitted = True

    def _compute_loss(
        self,
        seq: torch.Tensor,
        T_tok: torch.Tensor,
        X_float: torch.Tensor,
        T_flat: torch.Tensor,
    ) -> torch.Tensor:
        """Mean CE loss across treatment dimensions, plus optional balance/norm reg."""
        logits = self(seq)  # (n, seq_len, vocab_size)
        loss = torch.tensor(0.0, device=seq.device)
        for j in range(len(self._treatment_sizes)):
            pos = self._d_x - 1 + j
            loss = loss + F.cross_entropy(logits[:, pos, :], T_tok[:, j])
        loss = loss / len(self._treatment_sizes)

        if self._criterion is not None:
            log_joint = self._factored_joint(logits)
            loss = loss + self._criterion(log_joint, T_flat, X_float)

        return loss

    def _factored_joint(self, logits: torch.Tensor) -> torch.Tensor:
        """(n, M) log-joint from teacher-forced logits, summing log-probs across dims."""
        n = logits.shape[0]
        log_joint = torch.zeros(n, self._M, device=logits.device)
        for j, t_size in enumerate(self._treatment_sizes):
            pos = self._d_x - 1 + j
            t_start = self._sum_feat + int(self._t_offsets[j])
            log_probs_j = F.log_softmax(
                logits[:, pos, t_start:t_start + t_size], dim=-1
            )  # (n, t_size)
            log_joint = log_joint + log_probs_j[:, self._all_t_multi_tensor[:, j]]
        return log_joint

    def predict_proba(self, X: np.ndarray, M: int) -> np.ndarray:
        """Returns (n, M) propensity scores p(T | X)."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_tokens = X.astype(int) + self._x_offsets[np.newaxis, :]
        x_tensor = torch.LongTensor(X_tokens).to(self.device)
        self.eval()
        with torch.no_grad():
            if len(self._treatment_sizes) == 1:
                logits = self(x_tensor)
                t_logits = logits[:, -1, self._sum_feat:self._sum_feat + self._M]
                return F.softmax(t_logits, dim=-1).cpu().numpy()
            return self._predict_autoregressive(x_tensor)

    def _predict_autoregressive(self, X_tokens: torch.Tensor) -> np.ndarray:
        """
        Compute p(T|X) = prod_j p(t_j | X, t_0, ..., t_{j-1}) via d_t forward passes.

        Returns (n, M) probabilities in row-major order over treatment_sizes
        (same flat ordering as the dataset: last dimension varies fastest).
        """
        n = X_tokens.shape[0]

        log_joint = torch.zeros(n, 1, device=X_tokens.device)
        prefix_seqs = X_tokens.clone()  # (n, d_x)
        n_combos = 1

        for j, t_size in enumerate(self._treatment_sizes):
            old_n_combos = n_combos
            t_offset_j = int(self._t_offsets[j])

            # Forward: (n * old_n_combos, seq_len) → (n * old_n_combos, seq_len, vocab_size)
            logits = self(prefix_seqs)
            t_start = self._sum_feat + t_offset_j
            t_logits = logits[:, -1, t_start:t_start + t_size]  # (n * old_n_combos, t_size)
            log_probs = F.log_softmax(t_logits, dim=-1)

            # Update joint: (n, old_n_combos) → (n, old_n_combos * t_size)
            log_joint = log_joint.unsqueeze(-1) + log_probs.view(n, old_n_combos, t_size)
            n_combos = old_n_combos * t_size
            log_joint = log_joint.view(n, n_combos)

            # Build extended prefix for next iteration:
            # each of n * old_n_combos rows is replicated t_size times,
            # with a different t_j token appended each time.
            new_prefixes = (prefix_seqs.unsqueeze(1)
                            .expand(-1, t_size, -1)
                            .reshape(n * n_combos, -1))
            t_vals = torch.arange(t_size, device=X_tokens.device) + self._sum_feat + t_offset_j
            t_col = (t_vals.unsqueeze(0)
                     .expand(n * old_n_combos, -1)
                     .reshape(n * n_combos, 1))
            prefix_seqs = torch.cat([new_prefixes, t_col], dim=1)

        return torch.exp(log_joint).cpu().numpy()

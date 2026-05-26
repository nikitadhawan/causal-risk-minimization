"""Transformer APO model: g(t) mapping treatment vectors to outcomes."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from .transformer import TransformerBlock
from ..utils import get_device


class TransformerAPOModel(nn.Module):
    """
    Transformer-based APO model g(t): treatment vector → scalar outcome.

    Maps a multi-dimensional discrete treatment (t_0, ..., t_{d_t-1}) to an
    estimated average potential outcome.

    Vocabulary layout:
      T dimension j: [sum(treatment_sizes[:j]), sum(treatment_sizes[:j+1]))

    Input sequence:  [t_0_token, t_1_token, ..., t_{d_t-1}_token]
    Architecture uses non-causal (bidirectional) self-attention because the
    full treatment vector is available at inference time.

    loss_type='mse':
        Regression head (scalar). Trained with MSE on stabilized-weighted
        outcomes w = Y * p(T) / p(T|X).

    loss_type='ce':
        Binary head (2 logits). Trained with soft-label cross-entropy where
        the target for each sample is [1-w, w] with w = Y * p(T) / p(T|X).
        APO estimate = softmax(logits)[:, 1].

    Architecture is built lazily in fit() once treatment_sizes are known,
    so the model can be instantiated from config before the dataset is loaded.
    """

    def __init__(
        self,
        d_model: int = 8,
        num_heads: int = 2,
        d_hidden: int = 16,
        num_blocks: int = 2,
        lr: float = 0.001,
        epochs: int = 1000,
        batch_size: int = 256,
        use_minibatch: bool = False,
        verbose: bool = False,
        loss_type: str = 'mse',
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
        self.loss_type = loss_type
        self.device = get_device()
        self._is_fitted = False
        # Set in _build()
        self._treatment_sizes: list = None
        self._t_offsets: np.ndarray = None
        self._sum_t: int = None
        # Layers built in _build()
        self.embedding: nn.Embedding = None
        self.blocks: nn.ModuleList = None
        self.head: nn.Linear = None

    def _build(self, treatment_sizes: list) -> None:
        """Build embedding, blocks, and head once treatment_sizes are known."""
        self._treatment_sizes = treatment_sizes
        self._t_offsets = np.concatenate([[0], np.cumsum(treatment_sizes[:-1])])
        self._sum_t = sum(treatment_sizes)

        self.embedding = nn.Embedding(self._sum_t, self.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(self.d_model, self.num_heads, self.d_hidden, causal=False)
            for _ in range(self.num_blocks)
        ])
        out_dim = 2 if self.loss_type == 'ce' else 1
        self.head = nn.Linear(self.d_model, out_dim)
        self.to(self.device)

    def forward(self, t_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t_tokens: (batch, d_t) LongTensor of treatment tokens
        Returns:
            loss_type='mse': (batch,) scalars
            loss_type='ce':  (batch, 2) logits
        """
        x = self.embedding(t_tokens)          # (batch, d_t, d_model)
        for block in self.blocks:
            x = block(x)
        if self.loss_type == 'ce':
            repr = x[:, -1, :]               # (batch, d_model) — last token
        else:
            repr = x.mean(dim=1)             # (batch, d_model) — mean pool
        out = self.head(repr)                # (batch, out_dim)
        return out.squeeze(-1) if self.loss_type == 'mse' else out

    def _tokenize(self, T_multi: np.ndarray) -> torch.Tensor:
        """(n, d_t) int array → (n, d_t) LongTensor of vocabulary tokens."""
        tokens = T_multi.astype(int) + self._t_offsets[np.newaxis, :]
        return torch.LongTensor(tokens)

    def _compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == 'mse':
            return F.mse_loss(logits, target)
        else:
            # target: (batch,) values w = Y * weight in [0, 1]
            # soft label: [[1-w, w]] — cross-entropy with soft targets
            soft = torch.stack([1.0 - target, target], dim=1)  # (batch, 2)
            return -(soft * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

    def fit(
        self,
        T_multi: np.ndarray,
        Y: np.ndarray,
        M: int,
        val_T_multi: np.ndarray,
        val_Y: np.ndarray,
        weights: np.ndarray = None,
        val_weights: np.ndarray = None,
        treatment_sizes: list = None,
        **kwargs,
    ) -> None:
        """
        Train the APO model.

        Args:
            T_multi:         (n, d_t) integer treatment vectors
            Y:               (n,) observed outcomes (binary for loss_type='ce')
            M:               Number of treatments (kept for interface compat)
            val_T_multi:     (m, d_t) validation treatment vectors
            val_Y:           (m,) validation outcomes
            weights:         (n,) weights p(T)/p(T|X). If None,
                             Y is used as-is (treated as pre-weighted).
            val_weights:     (m,) validation weights
            treatment_sizes: Cardinalities per T dimension. Inferred as [M]
                             if not provided.
        """
        if treatment_sizes is None:
            treatment_sizes = [M]
        self._build(treatment_sizes)

        # w = Y * weight 
        w = Y.astype(float) if weights is None else (Y * weights).astype(float)

        t_tensor = self._tokenize(T_multi)
        w_tensor = torch.FloatTensor(w)

        val_w = val_Y.astype(float) if val_weights is None else (val_Y * val_weights).astype(float)
        val_tok = self._tokenize(val_T_multi)
        val_w_tensor = torch.FloatTensor(val_w)

        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        print_freq = max(1, self.epochs // 10)

        if self.use_minibatch:
            ds = TensorDataset(t_tensor, w_tensor)
            dataloader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(val_tok, val_w_tensor), batch_size=self.batch_size, shuffle=False)
            for epoch in range(self.epochs):
                self.train()
                total_loss = 0.0
                for t_batch, w_batch in dataloader:
                    t_batch = t_batch.to(self.device)
                    w_batch = w_batch.to(self.device)
                    optimizer.zero_grad()
                    loss = self._compute_loss(self(t_batch), w_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                if self.verbose and (epoch + 1) % print_freq == 0:
                    avg_loss = total_loss / len(dataloader)
                    self.eval()
                    val_total = 0.0
                    with torch.no_grad():
                        for vt, vw in val_loader:
                            vt, vw = vt.to(self.device), vw.to(self.device)
                            val_total += self._compute_loss(self(vt), vw).item()
                    print(f"  TransformerAPO epoch {epoch+1}/{self.epochs}, "
                            f"Loss: {avg_loss:.4f} (val: {val_total / len(val_loader):.4f})")
        else:
            t_tensor = t_tensor.to(self.device)
            w_tensor = w_tensor.to(self.device)
            for epoch in range(self.epochs):
                self.train()
                optimizer.zero_grad()
                loss = self._compute_loss(self(t_tensor), w_tensor)
                loss.backward()
                optimizer.step()
                if self.verbose and (epoch + 1) % print_freq == 0:
                    self.eval()
                    val_tok_t = val_tok.to(self.device)
                    val_w_t = val_w_tensor.to(self.device)
                    with torch.no_grad():
                        val_loss = self._compute_loss(self(val_tok_t), val_w_t).item()
                    print(f"  TransformerAPO epoch {epoch+1}/{self.epochs}, "
                            f"Loss: {loss.item():.4f} (val: {val_loss:.4f})")

        self._is_fitted = True

    def predict(self, T_multi: np.ndarray, M: int) -> np.ndarray:
        """
        Args:
            T_multi: (m, d_t) integer treatment vectors
            M:       Number of treatments (kept for interface compat)
        Returns:
            (m,) APO estimates
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        t_tensor = self._tokenize(T_multi).to(self.device)
        self.eval()
        with torch.no_grad():
            out = self(t_tensor)
            if self.loss_type == 'ce':
                return F.softmax(out, dim=-1)[:, 1].cpu().numpy()
            return out.cpu().numpy()

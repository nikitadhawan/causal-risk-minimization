"""Transformer outcome model: μ(x, t) mapping (confounders, treatment) → scalar outcome."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from .transformer import TransformerBlock
from ..utils import get_device


class TransformerOutcomeModel(nn.Module):
    """
    Transformer-based outcome model μ(x, t): (confounders, treatment) → scalar outcome.

    Both X and T are tokenized as discrete sequences and embedded via vocabulary
    lookups (same layout as TransformerAPOModel/TransformerPropensityModel).

    Vocabulary layout:
      X dimension j: [sum(feature_sizes[:j]),   sum(feature_sizes[:j+1]))
      T dimension j: [sum(treatment_sizes[:j]), sum(treatment_sizes[:j+1]))

    Input sequence: [x_0_token, ..., x_{d_x-1}_token, t_0_token, ..., t_{d_t-1}_token]

    Both loss types use bidirectional self-attention.

    loss_type='mse':
        Regression head (scalar). Mean-pooled sequence representation.
        Trained with MSE on observed outcomes Y.

    loss_type='ce':
        Binary head (2 logits). Last-token sequence representation.
        Trained with soft-label cross-entropy where the target for each
        sample is [1-Y, Y]. Prediction = softmax(logits)[:, 1].

    Architecture is built lazily in fit() once input dimensions are known.
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
        self._feature_sizes: list = None
        self._x_offsets: np.ndarray = None
        self._treatment_sizes: list = None
        self._t_offsets: np.ndarray = None
        # Layers built in _build()
        self.embedding: nn.Embedding = None
        self.blocks: nn.ModuleList = None
        self.head: nn.Linear = None

    def _build(self, feature_sizes: list, treatment_sizes: list) -> None:
        """Build layers once input dimensions are known."""
        self._feature_sizes = feature_sizes
        self._x_offsets = np.concatenate([[0], np.cumsum(feature_sizes[:-1])])
        self._treatment_sizes = treatment_sizes
        t_start = sum(feature_sizes)
        self._t_offsets = t_start + np.concatenate([[0], np.cumsum(treatment_sizes[:-1])])
        vocab_size = sum(feature_sizes) + sum(treatment_sizes)

        self.embedding = nn.Embedding(vocab_size, self.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(self.d_model, self.num_heads, self.d_hidden, causal=False)
            for _ in range(self.num_blocks)
        ])
        out_dim = 2 if self.loss_type == 'ce' else 1
        self.head = nn.Linear(self.d_model, out_dim)
        self.to(self.device)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (batch, d_x + d_t) LongTensor of X and T tokens
        Returns:
            loss_type='mse': (batch,) scalars
            loss_type='ce':  (batch, 2) logits
        """
        seq = self.embedding(tokens)             # (batch, d_x + d_t, d_model)
        for block in self.blocks:
            seq = block(seq)
        if self.loss_type == 'ce':
            repr = seq[:, -1, :]                 # (batch, d_model) — last token
        else:
            repr = seq.mean(dim=1)               # (batch, d_model) — mean pool
        out = self.head(repr)                    # (batch, out_dim)
        return out.squeeze(-1) if self.loss_type == 'mse' else out

    def _compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == 'mse':
            return F.mse_loss(logits, target)
        else:
            soft = torch.stack([1.0 - target, target], dim=1)  # (batch, 2)
            return -(soft * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

    def _tokenize(self, X: np.ndarray, T_multi: np.ndarray) -> torch.Tensor:
        """(n, d_x) and (n, d_t) int arrays → (n, d_x + d_t) LongTensor of vocabulary tokens."""
        x_tokens = X.astype(int) + self._x_offsets[np.newaxis, :]
        t_tokens = T_multi.astype(int) + self._t_offsets[np.newaxis, :]
        return torch.LongTensor(np.concatenate([x_tokens, t_tokens], axis=1))

    def fit(
        self,
        X: np.ndarray,
        T_multi: np.ndarray,
        Y: np.ndarray,
        M: int,
        val_X: np.ndarray,
        val_T_multi: np.ndarray,
        val_Y: np.ndarray,
        feature_sizes: list = None,
        treatment_sizes: list = None,
        **kwargs,
    ) -> None:
        """
        Train the outcome model.

        Args:
            X:               (n, d_x) integer confounder vectors
            T_multi:         (n, d_t) integer treatment vectors
            Y:               (n,) observed outcomes
            M:               Number of treatments (kept for interface compat)
            val_X:           (m, d_x) validation confounders for train/val comparison
            val_T_multi:     (m, d_t) validation treatment vectors
            val_Y:           (m,) validation outcomes
            feature_sizes:   Cardinalities per X dimension. Inferred from data if None.
            treatment_sizes: Cardinalities per T dimension. Inferred from data if None.
        """
        if feature_sizes is None:
            feature_sizes = (X.max(axis=0) + 1).tolist()
        if treatment_sizes is None:
            treatment_sizes = (T_multi.max(axis=0) + 1).tolist()
        self._build(feature_sizes, treatment_sizes)

        tokens = self._tokenize(X, T_multi)
        y_tensor = torch.FloatTensor(Y.astype(float))
        val_tok = self._tokenize(val_X, val_T_multi)
        val_y = torch.FloatTensor(val_Y.astype(float))

        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        print_freq = max(1, self.epochs // 10)

        if self.use_minibatch:
            ds = TensorDataset(tokens, y_tensor)
            dataloader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(val_tok, val_y), batch_size=self.batch_size, shuffle=False)
            for epoch in range(self.epochs):
                self.train()
                total_loss = 0.0
                for tok_batch, y_batch in dataloader:
                    tok_batch = tok_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    optimizer.zero_grad()
                    loss = self._compute_loss(self(tok_batch), y_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                if self.verbose and (epoch + 1) % print_freq == 0:
                    avg_loss = total_loss / len(dataloader)
                    self.eval()
                    val_total = 0.0
                    with torch.no_grad():
                        for vt, vy in val_loader:
                            vt, vy = vt.to(self.device), vy.to(self.device)
                            val_total += self._compute_loss(self(vt), vy).item()
                    print(f"  TransformerOutcome epoch {epoch+1}/{self.epochs}, "
                          f"Loss: {avg_loss:.4f} (val: {val_total / len(val_loader):.4f})")
        else:
            tokens = tokens.to(self.device)
            y_tensor = y_tensor.to(self.device)
            for epoch in range(self.epochs):
                self.train()
                optimizer.zero_grad()
                loss = self._compute_loss(self(tokens), y_tensor)
                loss.backward()
                optimizer.step()
                if self.verbose and (epoch + 1) % print_freq == 0:
                    self.eval()
                    val_tok_t = val_tok.to(self.device)
                    val_y_t = val_y.to(self.device)
                    with torch.no_grad():
                        val_loss = self._compute_loss(self(val_tok_t), val_y_t).item()
                    print(f"  TransformerOutcome epoch {epoch+1}/{self.epochs}, "
                          f"Loss: {loss.item():.4f} (val: {val_loss:.4f})")

        self._is_fitted = True

    def predict(self, X: np.ndarray, T_multi: np.ndarray, M: int) -> np.ndarray:
        """
        Args:
            X:       (n, d_x) integer confounder vectors
            T_multi: (n, d_t) integer treatment vectors
            M:       Number of treatments (kept for interface compat)
        Returns:
            (n,) outcome predictions
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        tokens = self._tokenize(X, T_multi).to(self.device)
        self.eval()
        with torch.no_grad():
            out = self(tokens)
            if self.loss_type == 'ce':
                return F.softmax(out, dim=-1)[:, 1].cpu().numpy()
            return out.cpu().numpy()

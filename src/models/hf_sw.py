"""HuggingFace stabilized weight model: w(x, t) via pooled LM hidden states."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict

from ..utils import get_device
from .prompt_format import PromptFormat


class HFSWModel(nn.Module):
    """
    HuggingFace stabilized weight model.

    Encodes "[x_text]\\n[t_text]" with a pretrained LM, masked-mean-pools the
    hidden states over non-padding positions, then projects to a scalar
    w(x,t) = p(t)/p(t|x) via a linear head + softplus.

    Trained with the same balance + normalization loss as SWModel:
      balance_loss = sum_t || E_w[X | T=t] - E[X] ||^2
      norm_loss    = sum_t (E[w_i | T=t] - 1)^2

    Both losses are computed per minibatch (same approximation as HFPropensityModel).

    Interface:
        fit(X, T_multi, T, M, ...)
        predict_weights(X, T_multi) → (n,) stabilized weights
    """

    def __init__(
        self,
        model_name: str = 'gpt2',
        lr: float = 2e-5,
        epochs: int = 3,
        batch_size: int = 16,
        max_length: int = 512,
        verbose: bool = False,
        freeze_base: bool = False,
        balance_weight: float = 1.0,
        norm_weight: float = 1.0,
        balance_moments: int = 1,
    ):
        super().__init__()
        self.model_name = model_name
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_length = max_length
        self.verbose = verbose
        self.freeze_base = freeze_base
        self.balance_weight = balance_weight
        self.norm_weight = norm_weight
        self.balance_moments = balance_moments
        self.device = get_device()
        self._is_fitted = False
        self.tokenizer = None
        self.lm = None
        self.head: nn.Sequential = None
        self.prompt_format: PromptFormat = None

    def _build(self, hidden_size: int = None) -> None:
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.lm = AutoModel.from_pretrained(self.model_name)
        if self.freeze_base:
            for p in self.lm.parameters():
                p.requires_grad = False
        self.lm.to(self.device)
        hidden_size = self.lm.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softplus(),
        ).to(self.device)

    def _pool(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Take last non-padding token hidden state → (B, hidden_size)."""
        hidden = self.lm(input_ids, attention_mask=attention_mask).last_hidden_state
        return hidden[:, -1, :]

    def _forward_w(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass: tokens → scalar stabilized weight per sample. Returns (B,)."""
        pooled = self._pool(input_ids, attention_mask)
        return self.head(pooled.float()).squeeze(-1)

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

    def _encode_seqs(self, X: np.ndarray, T_multi: np.ndarray):
        """Tokenize (x, t) pairs into input_ids and attention_mask tensors."""
        x_texts = [self.prompt_format.x_to_text(int(row[0])) for row in X]
        t_texts = [str(row[0]) for row in T_multi]
        seqs = [self.prompt_format.propensity_seq(x, t) for x, t in zip(x_texts, t_texts)]
        enc = self.tokenizer(
            seqs, padding=True, truncation=True,
            max_length=self.max_length, return_tensors='pt',
        )
        return enc['input_ids'], enc['attention_mask']

    def fit(
        self,
        X: np.ndarray,
        T_multi: np.ndarray,
        T: np.ndarray,
        M: int,
        val_X: np.ndarray,
        val_T_multi: np.ndarray,
        val_T: np.ndarray,
        **kwargs,
    ) -> None:
        """
        Fine-tune the LM head for stabilized weight prediction.

        Args:
            X:           (n, 1) integer covariate values
            T_multi:     (n, 1) treatment text strings for training observations
            T:           (n,) flat treatment indices
            M:           Number of treatments
            val_X:       (m, 1) validation covariate values
            val_T_multi: (m, 1) validation treatment text strings
            val_T:       (m,) validation flat treatment indices
            kwargs:      Must include prompt_format and balance_moment_targets
        """
        if T_multi is None:
            raise ValueError("T_multi must be provided")
        assert kwargs.get('prompt_format') is not None, "prompt_format must be provided"
        self.prompt_format = kwargs['prompt_format']
        balance_moment_targets = kwargs['balance_moment_targets']
        moment_targets_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in balance_moment_targets.items()}

        self._build()

        if self.verbose:
            x_text = self.prompt_format.x_to_text(int(X[0, 0]))
            print("Example sequence:")
            print(self.prompt_format.propensity_seq(x_text, str(T_multi[0, 0])))

        input_ids, attention_mask = self._encode_seqs(X, T_multi)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        T_tensor = torch.tensor(T, dtype=torch.long)

        ds = TensorDataset(input_ids, attention_mask, X_tensor, T_tensor)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        val_ids, val_mask = self._encode_seqs(val_X, val_T_multi)
        val_X_tensor = torch.tensor(val_X, dtype=torch.float32)
        val_T_tensor = torch.tensor(val_T, dtype=torch.long)
        val_loader = DataLoader(
            TensorDataset(val_ids, val_mask, val_X_tensor, val_T_tensor),
            batch_size=self.batch_size, shuffle=False,
        )

        params = (
            [p for p in self.lm.parameters() if p.requires_grad]
            + list(self.head.parameters())
        )
        opt = optim.AdamW(params, lr=self.lr)
        print_freq = max(1, self.epochs // 10)

        for epoch in range(self.epochs):
            self.lm.train()
            self.head.train()
            total = 0.0
            for b_ids, b_mask, b_X, b_T in loader:
                b_ids = b_ids.to(self.device)
                b_mask = b_mask.to(self.device)
                b_X = b_X.to(self.device)
                b_T = b_T.to(self.device)
                opt.zero_grad()
                w = self._forward_w(b_ids, b_mask)
                loss = self._compute_loss(w, b_X, b_T, M, moment_targets=moment_targets_t)
                loss.backward()
                opt.step()
                total += loss.item()
            if self.verbose and (epoch + 1) % print_freq == 0:
                self.lm.eval()
                self.head.eval()
                val_total = 0.0
                with torch.no_grad():
                    for vb_ids, vb_mask, vb_X, vb_T in val_loader:
                        vb_ids = vb_ids.to(self.device)
                        vb_mask = vb_mask.to(self.device)
                        vb_X = vb_X.to(self.device)
                        vb_T = vb_T.to(self.device)
                        val_w = self._forward_w(vb_ids, vb_mask)
                        val_total += self._compute_loss(val_w, vb_X, vb_T, M, moment_targets=moment_targets_t).item()
                print(f"  HFSWModel epoch {epoch+1}/{self.epochs}, "
                      f"loss={total / len(loader):.4f} (val: {val_total / len(val_loader):.4f})")

        self._is_fitted = True

    def predict_weights(self, X: np.ndarray, T_multi: np.ndarray) -> np.ndarray:
        """
        Predict stabilized weights w(x,t) for each observation.

        Args:
            X:       (n, 1) integer covariate values
            T_multi: (n, 1) treatment text strings

        Returns:
            (n,) stabilized weights
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        x_texts = [self.prompt_format.x_to_text(int(row[0])) for row in X]
        t_texts = [str(row[0]) for row in T_multi]
        seqs = [self.prompt_format.propensity_seq(x, t) for x, t in zip(x_texts, t_texts)]

        weights = []
        self.lm.eval()
        self.head.eval()
        with torch.no_grad():
            for i in range(0, len(seqs), self.batch_size):
                batch = seqs[i:i + self.batch_size]
                enc = self.tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=self.max_length, return_tensors='pt',
                )
                b_ids = enc['input_ids'].to(self.device)
                b_mask = enc['attention_mask'].to(self.device)
                w = self._forward_w(b_ids, b_mask)
                weights.extend(w.cpu().tolist())

        return np.array(weights)

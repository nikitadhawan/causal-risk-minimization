"""HuggingFace causal LM APO model g(t): treatment text → APO."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from ..utils import get_device
from .prompt_format import PromptFormat


class HFAPOModel(nn.Module):
    """
    Causal LM APO model g(t) = p(Y=1 | t) under the IPW-reweighted distribution.

    Fine-tunes a causal LM on "[t_text]\nOutcome: {Y}" sequences with per-sample
    importance weights. Loss is computed only at the final Y token position using
    soft-label cross-entropy:
        loss_i = -(1-w_i) * log p(Y=0 | t_i) - w_i * log p(Y=1 | t_i)
    where w_i = Y_i * importance_weight_i.

    Uses left-padding so:
      training:  logits[:, -2, :] predicts the Y token (last position)
      inference: logits[:, -1, :] predicts the Y token

    Interface:
        fit(T_multi, Y, M, weights=None, ...)
        predict(T_multi, M) → (m,)
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
    ):
        super().__init__()
        self.model_name = model_name
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_length = max_length
        self.verbose = verbose
        self.freeze_base = freeze_base
        self.device = get_device()
        self._is_fitted = False
        self.tokenizer = None
        self.lm = None
        self._id0 = self._id1 = None
        self.prompt_format: PromptFormat = None

    def _build(self) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.lm = AutoModelForCausalLM.from_pretrained(self.model_name)
        if self.freeze_base:
            for p in self.lm.parameters():
                p.requires_grad = False
        self.lm.to(self.device)
        ids0 = self.tokenizer.encode(self.prompt_format.neg_token, add_special_tokens=False)
        ids1 = self.tokenizer.encode(self.prompt_format.pos_token, add_special_tokens=False)
        assert len(ids0) == 1, f"neg_token {self.prompt_format.neg_token!r} tokenizes to {len(ids0)} tokens; must be 1"
        assert len(ids1) == 1, f"pos_token {self.prompt_format.pos_token!r} tokenizes to {len(ids1)} tokens; must be 1"
        self._id0 = ids0[0]
        self._id1 = ids1[0]

    def _make_seqs(self, T_multi: np.ndarray, Y: np.ndarray = None) -> list:
        return [
            self.prompt_format.apo_seq(str(t[0]), Y[i] if Y is not None else None)
            for i, t in enumerate(T_multi)
        ]

    def _compute_token_loss(self, logits: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Soft-label CE on the Y token position.

        Args:
            logits: (B, L, V) — full sequence logits with Y token appended (left-padded),
                    so logits[:, -2, :] predicts the Y token.
            w:      (B,) IPW-weighted soft targets in [0, 1]
        Returns:
            scalar loss
        """
        y_logits = logits[:, -2, :]                                      # (B, V)
        binary = torch.stack(
            [y_logits[:, self._id0], y_logits[:, self._id1]], dim=-1
        )                                                                 # (B, 2)
        soft = torch.stack([1.0 - w, w], dim=-1)                         # (B, 2)
        return -(soft * F.log_softmax(binary, dim=-1)).sum(dim=-1).mean()

    def fit(
        self,
        T_multi: np.ndarray,
        Y: np.ndarray,
        M: int,
        val_T_multi: np.ndarray,
        val_Y: np.ndarray,
        weights: np.ndarray = None,
        val_weights: np.ndarray = None,
        **kwargs,
    ) -> None:
        """
        Fine-tune with soft-label weighted loss on the Y token.

        Args:
            T_multi:     (n, 1) treatment text strings
            Y:           (n,) binary outcomes
            M:           Number of treatments (kept for interface compat)
            val_T_multi: (m, 1) validation treatment text strings
            val_Y:       (m,) validation binary outcomes
            weights:     (n,) importance weights p(T)/p(T|X). If None, Y is used as-is.
            val_weights: (m,) validation importance weights
        """
        if kwargs.get('prompt_format') is not None:
            self.prompt_format = kwargs['prompt_format']
        assert self.prompt_format is not None, "prompt_format must be provided by the dataset"
        self._build()
        w = Y.astype(float) if weights is None else (Y * weights).astype(float)
        val_w = val_Y.astype(float) if val_weights is None else (val_Y * val_weights).astype(float)

        seqs = self._make_seqs(T_multi, Y)
        if self.verbose:
            print("Example datapoint...")
            print(seqs[0])

        enc = self.tokenizer(seqs, padding=True, truncation=True,
                             max_length=self.max_length, return_tensors='pt')
        input_ids = enc['input_ids']        # (n, L)
        attention_mask = enc['attention_mask']
        w_tensor = torch.FloatTensor(w)
        ds = TensorDataset(input_ids, attention_mask, w_tensor)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        val_seqs = self._make_seqs(val_T_multi, val_Y)
        val_enc = self.tokenizer(val_seqs, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors='pt')
        val_w_tensor = torch.FloatTensor(val_w)
        val_loader = DataLoader(
            TensorDataset(val_enc['input_ids'], val_enc['attention_mask'], val_w_tensor),
            batch_size=self.batch_size, shuffle=False,
        )

        opt = optim.AdamW([p for p in self.lm.parameters() if p.requires_grad], lr=self.lr)
        print_freq = max(1, self.epochs // 10)

        for epoch in range(self.epochs):
            self.lm.train()
            total = 0.0
            for b_ids, b_mask, b_w in loader:
                b_ids = b_ids.to(self.device)
                b_mask = b_mask.to(self.device)
                b_w = b_w.to(self.device)
                opt.zero_grad()
                loss = self._compute_token_loss(
                    self.lm(b_ids, attention_mask=b_mask).logits, b_w
                )
                loss.backward()
                opt.step()
                total += loss.item()
            if self.verbose and (epoch + 1) % print_freq == 0:
                self.lm.eval()
                val_total = 0.0
                with torch.no_grad():
                    for vb_ids, vb_mask, vb_w in val_loader:
                        vb_ids = vb_ids.to(self.device)
                        vb_mask = vb_mask.to(self.device)
                        vb_w = vb_w.to(self.device)
                        val_total += self._compute_token_loss(
                            self.lm(vb_ids, attention_mask=vb_mask).logits, vb_w
                        ).item()
                print(f"  HFAPOModel epoch {epoch+1}/{self.epochs}, "
                      f"Loss: {total / len(loader):.4f} (val: {val_total / len(val_loader):.4f})")
        self._is_fitted = True

    def predict(self, T_multi: np.ndarray, M: int) -> np.ndarray:
        """
        Args:
            T_multi: (m, 1) treatment text strings
            M:       Number of treatments (kept for interface compat)
        Returns:
            (m,) APO estimates p(Y=1 | t)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        prompts = self._make_seqs(T_multi)
        enc = self.tokenizer(prompts, padding=True, truncation=True,
                             max_length=self.max_length, return_tensors='pt')
        input_ids = enc['input_ids']
        attention_mask = enc['attention_mask']

        self.lm.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(prompts), self.batch_size):
                b_ids = input_ids[i:i + self.batch_size].to(self.device)
                b_mask = attention_mask[i:i + self.batch_size].to(self.device)
                logits = self.lm(b_ids, attention_mask=b_mask).logits[:, -1, :]
                binary = torch.stack([logits[:, self._id0], logits[:, self._id1]], dim=-1)
                preds.append(F.softmax(binary.float(), dim=-1)[:, 1].cpu().numpy())
        return np.concatenate(preds)

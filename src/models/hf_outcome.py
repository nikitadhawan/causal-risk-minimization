"""HuggingFace causal LM outcome model: μ(x, t) = p(Y=1 | x, t)."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from ..utils import get_device
from .prompt_format import PromptFormat


class HFOutcomeModel(nn.Module):
    """
    Causal LM outcome model μ(x, t) = p(Y=1 | x, t).

    Fine-tunes a causal LM on sequences like "[x_text]\n[t_text]\nOutcome: {Y}",
    computing loss only on the final Y token (" No" or " Yes"). Prediction reads
    p(Y=1) from the softmax of the two outcome token logits at that position.

    Uses left-padding so logits[:, -2, :] always predicts the last (Y) token
    during training, and logits[:, -1, :] predicts it at inference time.

    Interface:
        fit(X, T_multi, Y, M, ...)
        predict(X, T_multi, M) → (n,)
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

    def _make_seqs(self, X: np.ndarray, T_multi: np.ndarray, Y: np.ndarray = None) -> list:
        """Build prompt strings, appending the outcome token when Y is provided."""
        return [
            self.prompt_format.outcome_seq(
                self.prompt_format.x_to_text(int(x[0])), str(t[0]),
                Y[i] if Y is not None else None,
            )
            for i, (x, t) in enumerate(zip(X, T_multi))
        ]

    def _make_labeled_tensors(self, X: np.ndarray, T_multi: np.ndarray, Y: np.ndarray):
        """Tokenize sequences and build label tensors with Y token labeled, rest -100."""
        seqs = self._make_seqs(X, T_multi, Y)
        enc = self.tokenizer(seqs, padding=True, truncation=True,
                             max_length=self.max_length, return_tensors='pt')
        input_ids = enc['input_ids']
        attention_mask = enc['attention_mask']
        labels = torch.full_like(input_ids, -100)
        labels[:, -1] = input_ids[:, -1]
        return input_ids, attention_mask, labels

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
        Fine-tune on "[x]\n[t]\nOutcome: {Y}" with loss on the Y token only.

        Args:
            X:           (n, 1) covariate strings
            T_multi:     (n, 1) treatment text strings
            Y:           (n,) binary outcomes
            M:           Number of treatments (kept for interface compat)
            val_X:       (m, 1) validation covariates
            val_T_multi: (m, 1) validation treatment text strings
            val_Y:       (m,) validation binary outcomes
        """
        assert kwargs.get('prompt_format') is not None, "prompt_format must be provided by the dataset"
        self.prompt_format = kwargs['prompt_format']
        self._build()

        if self.verbose:
            print("Example datapoint...")
            print(self._make_seqs(X[:1], T_multi[:1], Y[:1])[0])

        input_ids, attention_mask, labels = self._make_labeled_tensors(X, T_multi, Y)
        ds = TensorDataset(input_ids, attention_mask, labels)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        val_ids, val_mask, val_lbl = self._make_labeled_tensors(val_X, val_T_multi, val_Y)
        val_loader = DataLoader(
            TensorDataset(val_ids, val_mask, val_lbl),
            batch_size=self.batch_size, shuffle=False,
        )

        opt = optim.AdamW([p for p in self.lm.parameters() if p.requires_grad], lr=self.lr)
        print_freq = max(1, self.epochs // 10)

        for epoch in range(self.epochs):
            self.lm.train()
            total = 0.0
            for b_ids, b_mask, b_lbl in loader:
                b_ids, b_mask, b_lbl = b_ids.to(self.device), b_mask.to(self.device), b_lbl.to(self.device)
                opt.zero_grad()
                loss = self.lm(b_ids, attention_mask=b_mask, labels=b_lbl).loss
                loss.backward()
                opt.step()
                total += loss.item()
            if self.verbose and (epoch + 1) % print_freq == 0:
                self.lm.eval()
                val_total = 0.0
                with torch.no_grad():
                    for vb_ids, vb_mask, vb_lbl in val_loader:
                        vb_ids, vb_mask, vb_lbl = vb_ids.to(self.device), vb_mask.to(self.device), vb_lbl.to(self.device)
                        val_total += self.lm(vb_ids, attention_mask=vb_mask, labels=vb_lbl).loss.item()
                print(f"  HFOutcome epoch {epoch+1}/{self.epochs}, "
                      f"Loss: {total / len(loader):.4f} (val: {val_total / len(val_loader):.4f})")
        self._is_fitted = True

    def predict(self, X: np.ndarray, T_multi: np.ndarray, M: int) -> np.ndarray:
        """
        Args:
            X:       (n, 1) covariate strings
            T_multi: (n, 1) treatment text strings
            M:       Number of treatments (kept for interface compat)
        Returns:
            (n,) predicted p(Y=1 | x, t)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        prompts = self._make_seqs(X, T_multi)
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

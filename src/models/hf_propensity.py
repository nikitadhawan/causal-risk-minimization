"""HuggingFace causal LM propensity model p(t | x)."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from ..utils import get_device
from .prompt_format import PromptFormat


class HFPropensityModel(nn.Module):
    """
    Causal LM propensity model p(t | x).

    Fine-tunes a causal LM on "[x_text]\n[t_text]" sequences, computing loss
    only on the T token positions (X tokens are masked to -100). This directly
    trains the model to predict T given X.

    Scoring uses the LM's own token probabilities — no normalisation over the
    observed treatment set. This is correct because the LM distribution covers
    all possible token sequences, not just those seen during training.

    Interface:
        fit(X, T, M, T_multi=None, all_T_multi=None, ...)
        predict_proba(X, T_multi) → (n,)   p(T_i | X_i) for each pair
        marginal_proba(X)         → (M,)   E_X[p(T_j | X)] for all stored T_j
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
        balance_reg: float = 0.0,
        norm_reg: float = 0.0,
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
        self.balance_reg = balance_reg
        self.norm_reg = norm_reg
        self.balance_moments = balance_moments
        self.device = get_device()
        self._is_fitted = False
        self.tokenizer = None
        self.lm = None
        self.prompt_format: PromptFormat = None
        self._x_lens: dict = {}
        self._moment_targets: dict = None

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

    def _x_prefix_len(self, x_text: str) -> int:
        """Token count of the X prefix (used to mask X tokens from the training labels)."""
        if x_text not in self._x_lens:
            prefix = self.prompt_format.propensity_prefix(x_text)
            ids = self.tokenizer(prefix, add_special_tokens=False)['input_ids']
            self._x_lens[x_text] = len(ids)
        return self._x_lens[x_text]

    def _per_sample_logp(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        x_len: int,
    ) -> torch.Tensor:
        """
        Extract sum of log p over T tokens for each sample in a batch.

        All samples must share the same X prefix length (same x_text).
        Returns (B,) tensor, fully differentiable through logits.

        Args:
            logits:         (B, L, V) from LM forward pass
            input_ids:      (B, L) token IDs
            attention_mask: (B, L) with left-padding
            x_len:          number of tokens in the X prefix
        """
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (B, L-1, V)
        B, L = input_ids.shape

        per_logp = []
        for j in range(B):
            real_len = int(attention_mask[j].sum().item())
            t_len = real_len - x_len
            if t_len <= 0:
                raise ValueError(
                    f"T token length is {t_len} (x_len={x_len}, "
                    f"real_len={real_len}). No treatment tokens to score."
                )
            # T tokens at positions L-t_len to L-1 (left-padded).
            # Causal LM: logits[p] predicts token p+1.
            t_start = L - t_len
            t_tokens = input_ids[j, t_start:L]                  # (t_len,)
            lp = log_probs[j, t_start - 1:L - 1, :]             # (t_len, V)
            per_logp.append(lp[torch.arange(t_len, device=self.device), t_tokens].sum())
        return torch.stack(per_logp)  # (B,)

    def _tokenize_for_x(self, x_text: str, t_texts: list) -> dict:
        """Tokenize treatment texts paired with a specific X value."""
        seqs = [self.prompt_format.propensity_seq(x_text, t) for t in t_texts]
        enc = self.tokenizer(
            seqs, padding=True, truncation=True,
            max_length=self.max_length, return_tensors='pt',
        )
        return {
            'input_ids': enc['input_ids'].to(self.device),
            'attention_mask': enc['attention_mask'].to(self.device),
            'x_len': self._x_prefix_len(x_text),
        }

    def _make_labeled_tensors(self, X: np.ndarray, T_multi: np.ndarray):
        """Tokenize propensity sequences and build label tensors masking X prefix and PAD."""
        x_texts = [self.prompt_format.x_to_text(int(row[0])) for row in X]
        t_texts = [str(row[0]) for row in T_multi]
        seqs = [self.prompt_format.propensity_seq(x, t) for x, t in zip(x_texts, t_texts)]
        enc = self.tokenizer(seqs, padding=True, truncation=True,
                             max_length=self.max_length, return_tensors='pt')
        input_ids = enc['input_ids']
        attention_mask = enc['attention_mask']
        labels = input_ids.clone()
        labels[~attention_mask.bool()] = -100
        L = input_ids.shape[1]
        for i, x_text in enumerate(x_texts):
            x_len = self._x_prefix_len(x_text)
            real_len = int(attention_mask[i].sum().item())
            pad_len = L - real_len
            labels[i, pad_len:pad_len + x_len] = -100
        return input_ids, attention_mask, labels

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        M: int,
        val_X: np.ndarray,
        val_T_multi: np.ndarray,
        T_multi: np.ndarray = None,
        all_T_multi: np.ndarray = None,
        **kwargs,
    ) -> None:
        """
        Fine-tune on "[x]\n[t]" sequences with LM loss on T tokens only.

        Args:
            X:           (n, 1) integer covariate values
            T:           (n,) flat treatment indices (unused)
            M:           Number of treatments
            val_X:       (m, 1) validation covariate values
            val_T_multi: (m, 1) validation treatment texts
            T_multi:     (n, 1) treatment texts for training observations
            all_T_multi: (M, 1) treatment texts for all M treatments; used in
                         marginal_proba. Falls back to T_multi when not provided.
        """
        if T_multi is None:
            raise ValueError("T_multi must be provided")
        assert kwargs.get('prompt_format') is not None, "prompt_format must be provided by the dataset"
        self.prompt_format = kwargs['prompt_format']
        balance_moment_targets = kwargs['balance_moment_targets']
        self._moment_targets = {k: torch.tensor(v, dtype=torch.float32)
                                 for k, v in balance_moment_targets.items()}
        true_E_X = balance_moment_targets['mean']  # used for marginal p(x) weights
        self._unique_x_vals = np.unique(X[:, 0].astype(int)).tolist()
        self._p_x_weights = [
            float(true_E_X[0]) if x == 1 else float(1.0 - true_E_X[0])
            for x in self._unique_x_vals
        ]
        self._build()

        if self.verbose:
            x0 = self.prompt_format.x_to_text(int(X[0, 0]))
            print("Example datapoint...")
            print(self.prompt_format.propensity_seq(x0, str(T_multi[0, 0])))

        input_ids, attention_mask, labels = self._make_labeled_tensors(X, T_multi)
        self._t_texts_train = [str(row[0]) for row in T_multi]
        X_tensor = torch.tensor(X, dtype=torch.float32)
        indices = torch.arange(len(self._t_texts_train))
        ds = TensorDataset(indices, input_ids, attention_mask, labels, X_tensor)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        val_ids, val_mask, val_lbl = self._make_labeled_tensors(val_X, val_T_multi)
        val_t_texts = [str(row[0]) for row in val_T_multi]
        val_X_tensor = torch.tensor(val_X, dtype=torch.float32)
        val_indices = torch.arange(len(val_t_texts))
        val_loader = DataLoader(
            TensorDataset(val_indices, val_ids, val_mask, val_lbl, val_X_tensor),
            batch_size=self.batch_size, shuffle=False,
        )

        opt = optim.AdamW([p for p in self.lm.parameters() if p.requires_grad], lr=self.lr)
        print_freq = max(1, self.epochs // 10)
        use_reg = self.balance_reg > 0.0 or self.norm_reg > 0.0

        for epoch in range(self.epochs):
            self.lm.train()
            total = 0.0
            for b_idx, b_ids, b_mask, b_lbl, b_X in loader:
                b_ids = b_ids.to(self.device)
                b_mask = b_mask.to(self.device)
                b_lbl = b_lbl.to(self.device)
                opt.zero_grad()
                if use_reg:
                    b_t_texts = [self._t_texts_train[i] for i in b_idx.tolist()]
                    loss = self._lm_loss_with_reg(b_ids, b_mask, b_lbl, b_X.to(self.device), b_t_texts)
                else:
                    loss = self.lm(b_ids, attention_mask=b_mask, labels=b_lbl).loss
                loss.backward()
                opt.step()
                total += loss.item()
            if self.verbose and (epoch + 1) % print_freq == 0:
                self.lm.eval()
                val_total = 0.0
                with torch.no_grad():
                    for v_idx, vb_ids, vb_mask, vb_lbl, vb_X in val_loader:
                        vb_ids = vb_ids.to(self.device)
                        vb_mask = vb_mask.to(self.device)
                        vb_lbl = vb_lbl.to(self.device)
                        if use_reg:
                            vb_t_texts = [val_t_texts[i] for i in v_idx.tolist()]
                            val_total += self._lm_loss_with_reg(vb_ids, vb_mask, vb_lbl, vb_X.to(self.device), vb_t_texts).item()
                        else:
                            val_total += self.lm(vb_ids, attention_mask=vb_mask, labels=vb_lbl).loss.item()
                print(f"  HFPropensity epoch {epoch+1}/{self.epochs}, "
                      f"Loss: {total / len(loader):.4f} (val: {val_total / len(val_loader):.4f})")
        self._is_fitted = True

    def _compute_p_marg_batch(self, t_texts: list) -> torch.Tensor:
        """p_marg(T_i) = sum_x P(X=x) * p(T_i | X=x) for each T_i."""
        p_marg = torch.zeros(len(t_texts), device=self.device)
        for x_val, p_x in zip(self._unique_x_vals, self._p_x_weights):
            x_text = self.prompt_format.x_to_text(x_val)
            tok = self._tokenize_for_x(x_text, t_texts)
            logits = self.lm(tok['input_ids'], attention_mask=tok['attention_mask']).logits
            per_logp = self._per_sample_logp(logits, tok['input_ids'], tok['attention_mask'], tok['x_len'])
            p_marg = p_marg + p_x * per_logp.exp().clamp(min=1e-8)
        return p_marg

    def _lm_loss_with_reg(
        self,
        b_ids: torch.Tensor,
        b_mask: torch.Tensor,
        b_lbl: torch.Tensor,
        b_X: torch.Tensor,
        b_t_texts: list,
    ) -> torch.Tensor:
        """LM loss + balance/norm regularization.

        Balance: ||E_w[X] - E[X]||^2
        Norm:    (mean(w_i) - 1)^2    
        where w_i = p_marg(T_i) / p(T_i|X_i)
        """
        out = self.lm(b_ids, attention_mask=b_mask)
        logits = out.logits  # (B, L, V)
        V = logits.shape[-1]

        # Standard next-token LM loss over T positions
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = b_lbl[:, 1:].contiguous()
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, V), shift_labels.view(-1), ignore_index=-100
        )

        # Per-sample p(t_i | x_i), differentiable
        # Need x_len per sample — group by unique x_text in the batch
        B = b_ids.shape[0]
        per_logp_obs = torch.zeros(B, device=self.device)
        x_texts_batch = []
        for j in range(B):
            x_val = int(b_X[j, 0].item())
            x_texts_batch.append(self.prompt_format.x_to_text(x_val))

        for x_text in set(x_texts_batch):
            x_len = self._x_prefix_len(x_text)
            idx = [j for j, xt in enumerate(x_texts_batch) if xt == x_text]
            for j in idx:
                # Extract from the already-computed logits
                per_logp_obs[j] = self._per_sample_logp(
                    logits[j:j+1], b_ids[j:j+1], b_mask[j:j+1], x_len,
                )[0]

        p_obs = per_logp_obs.exp().clamp(min=1e-8)  # p(t_i | x_i)
        p_marg = self._compute_p_marg_batch(b_t_texts)  # p_T(t_i)
        w = p_marg / p_obs

        reg = torch.tensor(0.0, device=b_ids.device)

        if self.balance_reg > 0.0:
            mean = self._moment_targets['mean'].to(b_ids.device)
            std  = self._moment_targets['std'].to(b_ids.device)
            Z = (b_X - mean) / std
            balance_loss = torch.tensor(0.0, device=b_ids.device)
            for j in range(1, self.balance_moments + 1):
                E_wZj = (w.unsqueeze(1) * Z ** j).mean(0)
                target = self._moment_targets[j].to(b_ids.device)
                balance_loss = balance_loss + ((E_wZj - target) ** 2).mean() / (2 ** j)
            reg = reg + self.balance_reg * balance_loss

        if self.norm_reg > 0.0:
            reg = reg + self.norm_reg * (w.mean() - 1.0) ** 2

        return lm_loss + reg

    def _score_seqs(self, x_text: str, seqs: list) -> np.ndarray:
        """
        Sum log p of T tokens for each sequence in `seqs`.

        All sequences share the same X prefix (same x_text), so x_len is
        fixed across the batch. Returns (len(seqs),) log-probability scores.
        """
        x_len = self._x_prefix_len(x_text)

        scores = []
        with torch.no_grad():
            for i in range(0, len(seqs), self.batch_size):
                batch = seqs[i:i + self.batch_size]
                enc = self.tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=self.max_length, return_tensors='pt',
                )
                b_ids = enc['input_ids'].to(self.device)
                b_mask = enc['attention_mask'].to(self.device)

                logits = self.lm(b_ids, attention_mask=b_mask).logits
                per_logp = self._per_sample_logp(logits, b_ids, b_mask, x_len)
                scores.extend(per_logp.cpu().tolist())
        return np.array(scores)

    def predict_proba(self, X: np.ndarray, T_multi: np.ndarray) -> np.ndarray:
        """
        Compute p(T_i | X_i) for each observation pair.

        Groups observations by unique X value for efficient batching —
        O(|unique X| * n / batch_size) forward passes total.

        Args:
            X:       (n, 1) integer covariate values
            T_multi: (n, 1) treatment text strings

        Returns:
            (n,) array of p(T_i | X_i)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        t_texts = [str(row[0]) for row in T_multi]
        scores = np.zeros(len(X))

        self.lm.eval()
        unique_x_vals = list(dict.fromkeys(int(row[0]) for row in X))
        for x_val in unique_x_vals:
            x_text = self.prompt_format.x_to_text(x_val)
            idx = [i for i, row in enumerate(X) if int(row[0]) == x_val]
            seqs = [self.prompt_format.propensity_seq(x_text, t_texts[i]) for i in idx]
            log_scores = self._score_seqs(x_text, seqs)
            for i, ls in zip(idx, log_scores):
                scores[i] = np.exp(ls)

        return scores

    def marginal_proba(
        self,
        X: np.ndarray,
        T_multi: np.ndarray,
        true_E_X: np.ndarray,
    ) -> np.ndarray:
        """
        Compute marginal p(T_i) = E_X[p(T_i | X)] for each requested treatment.

        O(|unique X| * n / batch_size) forward passes.

        Args:
            X:        (n, 1) integer covariate values
            T_multi:  (n, 1) treatment text strings to compute marginals for
            true_E_X: (d_x,) true E[X]; for binary X, true_E_X[j] = P(X_j=1).

        Returns:
            (n,) marginal probability for each treatment in T_multi
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        t_texts = [str(row[0]) for row in T_multi]
        self.lm.eval()
        unique_x_vals = np.unique(X[:, 0].astype(int))
        p_x = np.array([
            float(true_E_X[0]) if x_val == 1 else float(1.0 - true_E_X[0])
            for x_val in unique_x_vals
        ])
        marginals = np.zeros(len(t_texts))
        for x_val, p in zip(unique_x_vals, p_x):
            x_text = self.prompt_format.x_to_text(int(x_val))
            seqs = [self.prompt_format.propensity_seq(x_text, t) for t in t_texts]
            marginals += p * np.exp(self._score_seqs(x_text, seqs))
        return marginals
"""
Semi-synthetic Amazon Electronics Reviews dataset.

  X = rating count covariate binned into num_bins levels (2, 4, or 8)
  T = review text — each review is a unique treatment (T[i] = i, so M = n)
  Y = binary purchase outcome sampled from LLM-generated probabilities
  true_apos = empirically-weighted sum_k P(X=k)*cond_outcome_xK per observation
"""

from pathlib import Path

import numpy as np
import pandas as pd

from typing import Dict

from ..models.prompt_format import PromptFormat
from ..utils import compute_x_moments


RATING_COUNT_THRESHOLDS = {
    2: [5000],
    4: [50, 1000, 20000],
    8: [10, 50, 200, 1000, 5000, 20000, 100000],
}

_T_COL = "t"
_APO_COL = "apo"
_Y_COL = "y_purchase"


def _assign_bin(row: dict, thresholds: list[int]) -> int:
    """Return the bin index for a row's x_rating_count given a list of thresholds."""
    val = float(row.get("x_rating_count", 0) or 0)
    for i, t in enumerate(thresholds):
        if val <= t:
            return i
    return len(thresholds)


def _bin_label(bin_idx: int, thresholds: list[int]) -> str:
    """Return a human-readable range string for bin_idx, e.g. '51–200'."""
    if bin_idx == 0:
        return f"1\u2013{thresholds[0]:,}"
    elif bin_idx == len(thresholds):
        return f"more than {thresholds[-1]:,}"
    else:
        return f"{thresholds[bin_idx - 1] + 1:,}\u2013{thresholds[bin_idx]:,}"


_ALL_X_COLS = ["x_rating_count"]


class AmazonReviewsPromptFormat(PromptFormat):
    """Task-specific prompt format for Amazon purchase prediction.

    X is the product popularity tier (rating count bin); T is the product review
    text; Y is whether the customer purchases after reading the review.
    """

    def __init__(self, thresholds: list = None):
        self.pos_token = " Yes"
        self.neg_token = " No"
        self.thresholds = thresholds if thresholds is not None else [5000]

    def x_to_text(self, x_val: int) -> str:
        return f"Number of Ratings: {_bin_label(x_val, self.thresholds)}"

    def outcome_seq(self, x_text: str, t_text: str, y: int = None) -> str:
        s = f"A potential customer sees the following information about a product.\n\n{x_text}\n\nA user's feedback:\n{t_text}\n\nWill the customer purchase this product?\nAnswer:"
        if y is not None:
            s += self.pos_token if int(y) == 1 else self.neg_token
        return s

    def propensity_prefix(self, x_text: str) -> str:
        return f"The following is some information about a product:\n{x_text}\n\nA user's feedback on the product:\n"

    def apo_seq(self, t_text: str, y: int = None) -> str:
        s = f"A potential customer sees the following information about a product.\n\nA user's feedback:\n{t_text}\n\nWill the customer purchase this product?\nAnswer:"
        if y is not None:
            s += self.pos_token if int(y) == 1 else self.neg_token
        return s


class AmazonReviews:
    """
    Amazon Electronics Reviews dataset for observational causal inference.

    Each row is one observation. The treatment T is the review text, so every
    observation has a unique treatment (M = n). The covariate X is the rating
    count bin index in {0, ..., num_bins-1}. The binary outcome Y is whether a
    customer would purchase after reading the review.

    Attributes set after __init__:
      M            : int — number of treatments (= n, one per review)
    """

    def __init__(
        self,
        category: str = "Electronics",
        n: int = 5000,  # used to construct csv_path if not provided
        num_bins: int = 2,
        csv_path: str | Path = None,
        train_frac: float = 0.6,
        val_frac: float = 0.2,
        seed: int = 42,
    ):
        if num_bins not in RATING_COUNT_THRESHOLDS:
            raise ValueError(f"num_bins must be one of {sorted(RATING_COUNT_THRESHOLDS)}, got {num_bins}")
        self.category = category
        self.n = n
        self.num_bins = num_bins
        self._thresholds = RATING_COUNT_THRESHOLDS[num_bins]
        self.csv_path = Path(csv_path)
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.seed = seed
        self._load()

    # ── Loading ────────────────────────────────────────────────────────────

    def _load(self):
        df = pd.read_csv(self.csv_path)

        assert self.n <= len(df), f"Requested n={self.n} exceeds dataset size {len(df)}"
        if self.n < len(df):
            df = df.sample(n=self.n, random_state=self.seed).reset_index(drop=True)

        self.M = self.n   # each review is a unique treatment

        self._X = np.array([                            # (n, 1) int in {0, ..., num_bins-1}
            _assign_bin(row, self._thresholds) for row in df.to_dict("records")
        ], dtype=int).reshape(-1, 1)
        self._T_flat = np.arange(self.n, dtype=int)    # T[i] = i
        self._T_text = df[_T_COL].to_numpy(dtype=str)  # (n,)
        self._Y = df[_Y_COL].to_numpy(dtype=int)        # (n,)
        self._true_apos = df[_APO_COL].to_numpy(dtype=float)  # (n,)

        # -- Train / val / test split --------------------------------------
        rng = np.random.RandomState(self.seed)
        perm = rng.permutation(self.n)
        n_train = max(1, int(np.floor(self.n * self.train_frac)))
        n_val = max(1, int(np.floor(self.n * self.val_frac)))
        if n_train + n_val >= self.n:
            n_val = self.n - n_train - 1

        self._train_idx = np.sort(perm[:n_train])
        self._val_idx = np.sort(perm[n_train:n_train + n_val])
        self._test_idx = np.sort(perm[n_train + n_val:])

    # ── Full dataset ──────────────────────────────────────────────────────

    @property
    def X(self) -> np.ndarray:
        """(n, 1) rating count bin index in {0, ..., num_bins-1}."""
        return self._X

    @property
    def T(self) -> np.ndarray:
        """(n,) flat treatment indices in {0, ..., M-1} (T[i] = i)."""
        return self._T_flat

    @property
    def T_multi(self) -> np.ndarray:
        """(n, 1) treatment feature — the review text for each observation."""
        return self._T_text.reshape(-1, 1)

    @property
    def all_T_multi(self) -> np.ndarray:
        """(M, 1) review text for each of the M treatments."""
        return self._T_text.reshape(-1, 1)

    @property
    def Y(self) -> np.ndarray:
        """(n,) binary purchase outcomes."""
        return self._Y

    # ── Splits ────────────────────────────────────────────────────────────

    @property
    def X_train(self) -> np.ndarray:
        return self._X[self._train_idx]

    @property
    def X_val(self) -> np.ndarray:
        return self._X[self._val_idx]

    @property
    def X_test(self) -> np.ndarray:
        return self._X[self._test_idx]

    @property
    def T_train(self) -> np.ndarray:
        return self._T_flat[self._train_idx]

    @property
    def T_multi_train(self) -> np.ndarray:
        return self._T_text[self._train_idx].reshape(-1, 1)

    @property
    def Y_train(self) -> np.ndarray:
        return self._Y[self._train_idx]

    @property
    def T_val(self) -> np.ndarray:
        return self._T_flat[self._val_idx]

    @property
    def T_multi_val(self) -> np.ndarray:
        return self._T_text[self._val_idx].reshape(-1, 1)

    @property
    def Y_val(self) -> np.ndarray:
        return self._Y[self._val_idx]

    @property
    def T_test(self) -> np.ndarray:
        return self._T_flat[self._test_idx]

    @property
    def T_multi_test(self) -> np.ndarray:
        return self._T_text[self._test_idx].reshape(-1, 1)

    @property
    def Y_test(self) -> np.ndarray:
        return self._Y[self._test_idx]

    # ── Counts ────────────────────────────────────────────────────────────

    @property
    def n_train(self) -> int:
        return len(self._train_idx)

    @property
    def n_val(self) -> int:
        return len(self._val_idx)

    @property
    def n_test(self) -> int:
        return len(self._test_idx)

    # ── Ground truth ──────────────────────────────────────────────────────

    @property
    def true_apos(self) -> np.ndarray:
        """(M,) true APO = sum_k P(X=k)*cond_outcome_xK per treatment."""
        return self._true_apos

    @property
    def true_E_X(self) -> np.ndarray:
        """(d_x,) empirical E[X] over the full dataset."""
        return self._X.mean(axis=0)

    def x_moments(self, k: int) -> Dict[int, np.ndarray]:
        """Empirical moments of X_train up to order k.

        Returns:
            Dict mapping j -> (d_x,) array of E[X^j] for j = 1..k
        """
        return compute_x_moments(self.X_train, k)

    def naive_apo_estimates(self) -> np.ndarray:
        """(M,) naive APO = Y[i] for each treatment (since each treatment is observed once)."""
        return self._Y.astype(float)

    def confounding_bias(self) -> np.ndarray:
        return self.naive_apo_estimates() - self.true_apos

    # ── Helpers ───────────────────────────────────────────────────────────

    @property
    def feature_sizes(self) -> list:
        """[num_bins] — X takes values in {0, ..., num_bins-1}."""
        return [self.num_bins]

    @property
    def treatment_sizes(self) -> list:
        """[M] — single flat treatment dimension of size M."""
        return [self.M]

    @property
    def prompt_format(self) -> AmazonReviewsPromptFormat:
        """Task-specific prompt format for HuggingFace causal LM models."""
        return AmazonReviewsPromptFormat(thresholds=self._thresholds)

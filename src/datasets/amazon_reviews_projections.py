"""
Semi-synthetic Amazon Electronics Reviews dataset for causal inference,
with configurable treatment type.

  X = rating count covariate binned into num_bins levels (2, 4, or 8)
  T = one of: review text ("t"), review rating ("rating"),
              review sentiment ("sentiment"), or review length ("length")
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

_APO_COL = "apo"
_Y_COL = "y_purchase"

_TREATMENT_COLS = {
    "rating":    "rating",
    "sentiment": "sentiment",
    "length":    "length",
}

# (label used in outcome_seq/apo_seq,  label used in propensity_prefix)
_TREATMENT_LABELS = {
    # used when treat_dim == "high" (full review text)
    "high": ("A user's feedback",            "A user's feedback on the product"),
    # used when treat_dim == "low" (projected treatment column)
    "rating":    ("A user's rating",              "A user's rating of the product"),
    "sentiment": ("Sentiment of a user's review", "Sentiment of a user's review of the product"),
    "length":    ("Length of a user's review",    "Length of a user's review of the product"),
}


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


class AmazonReviewsProjectionsPromptFormat(PromptFormat):
    """Task-specific prompt format for Amazon purchase prediction.

    X is the product popularity tier (rating count bin).
    T is configurable: review text, rating, sentiment, or review length.
    Y is whether the customer purchases after seeing the treatment.
    """

    def __init__(self, thresholds: list = None,
                 treatment_label: str = "A user's feedback",
                 propensity_label: str = "A user's feedback on the product"):
        self.pos_token = " Yes"
        self.neg_token = " No"
        self.thresholds = thresholds if thresholds is not None else [5000]
        self.treatment_label = treatment_label
        self.propensity_label = propensity_label

    def x_to_text(self, x_val: int) -> str:
        return f"Number of Ratings: {_bin_label(x_val, self.thresholds)}"

    def outcome_seq(self, x_text: str, t_text: str, y: int = None) -> str:
        s = (
            f"A potential customer sees the following information about a product.\n\n"
            f"{x_text}\n\n"
            f"{self.treatment_label}:\n{t_text}\n\n"
            f"Will the customer purchase this product?\nAnswer:"
        )
        if y is not None:
            s += self.pos_token if int(y) == 1 else self.neg_token
        return s

    def propensity_prefix(self, x_text: str) -> str:
        return (
            f"The following is some information about a product:\n{x_text}\n\n"
            f"{self.propensity_label}:\n"
        )

    def apo_seq(self, t_text: str, y: int = None) -> str:
        s = (
            f"A potential customer sees the following information about a product.\n\n"
            f"{self.treatment_label}:\n{t_text}\n\n"
            f"Will the customer purchase this product?\nAnswer:"
        )
        if y is not None:
            s += self.pos_token if int(y) == 1 else self.neg_token
        return s


class AmazonReviewsProjections:
    """
    Amazon Electronics Reviews dataset with configurable treatment type and dimensionality.

    Each row is one observation. The treatment attribute is set by treatment_type:
      "rating"    — rating in the review
      "sentiment" — sentiment of the review
      "length"    — length of the review

    The actual treatment observation column is controlled by treat_dim:
      "high"  — use the full review text ("t" column)
      "low"   — use the treatment_type column (rating / sentiment / length)

    The covariate X is the rating count bin index in {0, ..., num_bins-1}. 
    The binary outcome Y is whether a customer would purchase after reading the treatment.

    Attributes set after __init__:
      M            : int — number of treatments (= n, one per observation)
    """

    def __init__(
        self,
        category: str = "Electronics",
        n: int = 5000,
        num_bins: int = 2,
        csv_path: str | Path = None,
        train_frac: float = 0.6,
        val_frac: float = 0.2,
        seed: int = 42,
        treatment_type: str = "rating",
        treat_dim: str = "high",
    ):
        if num_bins not in RATING_COUNT_THRESHOLDS:
            raise ValueError(f"num_bins must be one of {sorted(RATING_COUNT_THRESHOLDS)}, got {num_bins}")
        if treatment_type not in _TREATMENT_COLS:
            raise ValueError(f"treatment_type must be one of {list(_TREATMENT_COLS)}, got {treatment_type!r}")
        if treat_dim not in ("high", "low"):
            raise ValueError(f"treat_dim must be 'high' or 'low', got {treat_dim!r}")
        self.category = category
        self.n = n
        self.num_bins = num_bins
        self._thresholds = RATING_COUNT_THRESHOLDS[num_bins]
        self.csv_path = Path(csv_path)
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.seed = seed
        self.treatment_type = treatment_type
        self.treat_dim = treat_dim
        self._load()

    # ── Loading ────────────────────────────────────────────────────────────

    def _load(self):
        df = pd.read_csv(self.csv_path)

        assert self.n <= len(df), f"Requested n={self.n} exceeds dataset size {len(df)}"
        if self.n < len(df):
            df = df.sample(n=self.n, random_state=self.seed).reset_index(drop=True)

        self._X = np.array([
            _assign_bin(row, self._thresholds) for row in df.to_dict("records")
        ], dtype=int).reshape(-1, 1)

        t_col = "t" if self.treat_dim == "high" else _TREATMENT_COLS[self.treatment_type]
        self._T_text = df[t_col].to_numpy(dtype=str)       # (n,)

        self._Y = df[_Y_COL].to_numpy(dtype=int)            # (n,)
        highdim_apos = df[_APO_COL].to_numpy(dtype=float)  # (n,) per-observation

        # Low-dim treatment values for each observation — used for projection.
        low_col = _TREATMENT_COLS[self.treatment_type]
        self._T_low_text = df[low_col].to_numpy(dtype=str)  # (n,)
        self.treatment_values = sorted(set(self._T_low_text))  # canonical ordering

        if self.treat_dim == "high":
            # Each observation has a unique treatment (the full review text).
            self.M = self.n
            self._T_flat = np.arange(self.n, dtype=int)
            self._all_T_text = self._T_text                 # (n,) — all M=n treatment texts
        else:
            # Multiple observations share the same low-dim treatment value.
            self.M = len(self.treatment_values)
            val_to_idx = {v: i for i, v in enumerate(self.treatment_values)}
            self._T_flat = np.array([val_to_idx[v] for v in self._T_low_text], dtype=int)
            self._all_T_text = np.array(self.treatment_values, dtype=str)  # (M,)

        self._true_apos = self.project_apos(highdim_apos, self._T_low_text)

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
        """(n, 1) treatment feature text for each observation."""
        return self._T_text.reshape(-1, 1)

    @property
    def all_T_multi(self) -> np.ndarray:
        """(M, 1) treatment feature text for each of the M treatments."""
        return self._all_T_text.reshape(-1, 1)

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
        """(n_unique,) projected APO — mean of per-obs APOs grouped by treatment_type value.
        Ordered by self.treatment_values (sorted). Lengths: 5 for rating, 2 for sentiment or length.
        """
        return self._true_apos

    @property
    def true_E_X(self) -> np.ndarray:
        """(d_x,) empirical E[X] over the full dataset."""
        return self._X.mean(axis=0)

    def x_moments(self, k: int) -> Dict[int, np.ndarray]:
        """Empirical moments of X_train up to order k."""
        return compute_x_moments(self.X_train, k)

    def project_apos(self, predicted_apos: np.ndarray, treatment_obs: np.ndarray) -> np.ndarray:
        """Project per-observation APOs to the low-dim treatment space.

        For each value in self.treatment_values, averages predicted_apos over
        all observations whose treatment_obs matches that value.

        Args:
            predicted_apos: (n,) array of per-observation APO values.
            treatment_obs:  (n,) array of low-dim treatment observations (str),
                            e.g. the rating / sentiment / length column values.

        Returns:
            (M,) array of projected APOs, ordered by self.treatment_values.
        """
        treatment_obs = np.asarray(treatment_obs, dtype=str)
        return np.array([
            predicted_apos[treatment_obs == v].mean() for v in self.treatment_values
        ], dtype=float)

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
    def prompt_format(self) -> AmazonReviewsProjectionsPromptFormat:
        """Task-specific prompt format, adapted to the active treatment type and dimensionality."""
        label_key = "high" if self.treat_dim == "high" else self.treatment_type
        t_label, p_label = _TREATMENT_LABELS[label_key]
        return AmazonReviewsProjectionsPromptFormat(
            thresholds=self._thresholds,
            treatment_label=t_label,
            propensity_label=p_label,
        )

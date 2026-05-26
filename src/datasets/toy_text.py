"""
Synthetic sequence dataset with discrete multi-dimensional covariates, discrete multi-dimensional treatments, and discrete outcomes.
"""

from typing import Dict
import numpy as np

from ..utils import compute_x_moments


class SyntheticTextDataset:
    """
    Synthetic dataset where each unit has multi-dimensional discrete covariates,
    a multi-dimensional discrete treatment, and a discrete outcome. All values
    map to unique integer tokens in a shared vocabulary for transformer input.

    Treatment splits partition flat treatment indices into non-overlapping
    train/val/test subsets.
    """

    FEATURE_SIZES = [5, 4, 2, 3]

    def __init__(
        self,
        treatment_sizes: tuple = (4, 3, 3),
        n_outcomes: int = 2,
        n: int = 20000,
        outcome_intercept: float = 0.0,
        outcome_coef_t: tuple = (0.3, 0.2, 0.15),
        outcome_coef_x: tuple = (0.15, 0.10, 0.25, 0.15),
        interaction_tt: float = 0.5,
        interaction_tx: float = 0.5,
        interaction_xx: float = 0.3,
        interaction_x2t: float = 1.0,
        interaction_x3t: float = 0.5,
        confound_strength: float = 5.0,
        confound_x2: float = 0.0,
        confound_x3: float = 0.0,
        train_frac: float = 0.6,
        val_frac: float = 0.2,
        seed: int = 42,
    ):
        self.treatment_sizes = list(treatment_sizes)
        self.M = int(np.prod(treatment_sizes))
        self.d_t = len(treatment_sizes)
        self.n_outcomes = n_outcomes
        self.n = n
        self.outcome_intercept = outcome_intercept
        self.outcome_coef_t = np.array(outcome_coef_t, dtype=float)
        self.outcome_coef_x = np.array(outcome_coef_x, dtype=float)
        self.interaction_tt = interaction_tt
        self.interaction_tx = interaction_tx
        self.interaction_xx = interaction_xx
        self.interaction_x2t = interaction_x2t
        self.interaction_x3t = interaction_x3t
        self.confound_strength = confound_strength
        self.confound_x2 = confound_x2
        self.confound_x3 = confound_x3
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.seed = seed

        assert self.M >= 3, "Total treatments must be >= 3 for non-overlapping splits"
        assert len(outcome_coef_t) == self.d_t
        assert len(outcome_coef_x) == len(self.FEATURE_SIZES)

        # Precompute (M, d_t) lookup: flat treatment index → multi-dim features
        grids = np.meshgrid(*[np.arange(v) for v in self.treatment_sizes], indexing="ij")
        self._all_t_multi = np.column_stack([g.ravel() for g in grids])

        self._generate()

    # ── Vocabulary metadata ────────────────────────────────────────────────

    @property
    def feature_sizes(self) -> list:
        """List of discrete cardinalities for each X feature."""
        return list(self.FEATURE_SIZES)

    @property
    def n_x_combos(self) -> int:
        return int(np.prod(self.FEATURE_SIZES))

    @property
    def n_t_combos(self) -> int:
        return self.M

    @property
    def x_vocab_offsets(self) -> list:
        offsets = [0]
        for v in self.FEATURE_SIZES[:-1]:
            offsets.append(offsets[-1] + v)
        return offsets

    @property
    def t_vocab_offsets(self) -> list:
        """Starting token index for each treatment dimension."""
        base = sum(self.FEATURE_SIZES)
        offsets = [base]
        for v in self.treatment_sizes[:-1]:
            offsets.append(offsets[-1] + v)
        return offsets

    @property
    def y_vocab_offset(self) -> int:
        return sum(self.FEATURE_SIZES) + sum(self.treatment_sizes)

    @property
    def vocab_size(self) -> int:
        return sum(self.FEATURE_SIZES) + sum(self.treatment_sizes) + self.n_outcomes

    @property
    def seq_len(self) -> int:
        """Length of (X, T) token sequence: d_x + d_t."""
        return len(self.FEATURE_SIZES) + self.d_t

    # ── Flat indexing helpers ──────────────────────────────────────────────

    def _flat_index(self, vals: np.ndarray, sizes: list) -> np.ndarray:
        """Convert (n, d) matrix of discrete values to (n,) flat indices."""
        idx = np.zeros(len(vals), dtype=int)
        stride = 1
        for j in range(len(sizes) - 1, -1, -1):
            idx += vals[:, j] * stride
            stride *= sizes[j]
        return idx

    def _flat_to_multi(self, flat_idx: int, sizes: list) -> np.ndarray:
        """Convert flat index to (d,) integer vector."""
        vals = []
        for v in reversed(sizes):
            vals.append(flat_idx % v)
            flat_idx //= v
        return np.array(list(reversed(vals)))

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    # ── Outcome model ─────────────────────────────────────────────────────

    def _outcome_latent(self, T_multi: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Compute E[Y_latent | T, X] (no noise).

        Args:
            T_multi: (n, d_t) multi-dimensional treatment features
            X: (n, d_x) covariate features
        """
        mu = (self.outcome_intercept
              + T_multi @ self.outcome_coef_t
              + X @ self.outcome_coef_x)

        t0_bar = T_multi[:, 0]
        t1_bar = T_multi[:, 1]
        mu += self.interaction_tt * t0_bar * t1_bar

        sev_bar = X[:, 0]
        mu += self.interaction_tx * t0_bar * sev_bar

        com_bar = X[:, 2]
        mu += self.interaction_xx * sev_bar * com_bar
        mu += self.interaction_x2t * t0_bar * sev_bar ** 2
        mu += self.interaction_x3t * t0_bar * sev_bar ** 3

        return mu

    # ── Propensity model ──────────────────────────────────────────────────

    def _build_propensity_table(self) -> np.ndarray:
        """
        Build (n_x_combos, M) joint propensity table.

        Factored: P(T|X) = P(drug|X) · P(dosage|X) · P(route|X), where:
          severity   → more aggressive drug  (higher sev → higher drug index)
          risk_score → conservative dosing    (higher risk → lower dosage)
          age_group  → route preference       (older → higher route index)
        """
        # (feature_index, sign): which X feature drives each T dim, and direction
        channels = [(0, +1.0), (3, -1.0), (1, +1.0)]
        if self.d_t > len(channels):
            channels += [(None, 0.0)] * (self.d_t - len(channels))

        # Build per-dimension conditional tables: (n_x_combos, treatment_sizes[j])
        dim_tables = []
        for j in range(self.d_t):
            feat_idx, sign = channels[j]
            table_j = np.zeros((self.n_x_combos, self.treatment_sizes[j]))
            t_bar = np.arange(self.treatment_sizes[j], dtype=float)

            for flat_x in range(self.n_x_combos):
                if feat_idx is not None:
                    x = self._flat_to_multi(flat_x, self.FEATURE_SIZES)
                    x_bar = float(x[feat_idx])
                    logits = (self.confound_strength * sign * x_bar * t_bar
                              + self.confound_x2 * sign * x_bar ** 2 * t_bar
                              + self.confound_x3 * sign * x_bar ** 3 * t_bar)
                else:
                    logits = np.zeros(self.treatment_sizes[j])
                logits -= logits.max()
                table_j[flat_x] = np.exp(logits) / np.exp(logits).sum()

            dim_tables.append(table_j)

        # Joint table: outer product of per-dimension probabilities
        joint_table = np.ones((self.n_x_combos, self.M))
        for j in range(self.d_t):
            joint_table *= dim_tables[j][:, self._all_t_multi[:, j]]

        self._dim_tables = dim_tables
        return joint_table

    # ── Ground truth APOs ─────────────────────────────────────────────────

    def _compute_true_apos(self) -> np.ndarray:
        """Compute E[Y(t)] = E_X[σ(μ(t, X))] by enumeration over all X combos."""
        all_x = np.array([
            self._flat_to_multi(i, self.FEATURE_SIZES)
            for i in range(self.n_x_combos)
        ])
        true_apos = np.zeros(self.M)
        for t_flat in range(self.M):
            t_multi = np.tile(self._all_t_multi[t_flat], (self.n_x_combos, 1))
            mu = self._outcome_latent(t_multi, all_x)
            true_apos[t_flat] = self._sigmoid(mu).mean()
        return true_apos

    # ── Treatment splits ──────────────────────────────────────────────────

    def _split_treatments(self, rng) -> tuple:
        perm = rng.permutation(self.M)
        n_train = max(1, int(np.floor(self.M * self.train_frac)))
        n_val = max(1, int(np.floor(self.M * self.val_frac)))
        if self.M - n_train - n_val < 1:
            n_val = self.M - n_train - 1
        return (perm[:n_train],
                perm[n_train:n_train + n_val],
                perm[n_train + n_val:])

    # ── Main generation ───────────────────────────────────────────────────

    def _generate(self):
        rng = np.random.RandomState(self.seed)

        self._propensity_table = self._build_propensity_table()

        # Sample X
        self._X_features = np.column_stack([
            rng.choice(v, size=self.n) for v in self.FEATURE_SIZES
        ])
        self._X_flat = self._flat_index(self._X_features, self.FEATURE_SIZES)

        # Sample T | X via inverse CDF
        probs = self._propensity_table[self._X_flat]
        u = rng.rand(self.n, 1)
        self._T_flat = (u < probs.cumsum(axis=1)).argmax(axis=1)
        self._T_multi = self._all_t_multi[self._T_flat]

        self._propensity_scores = self._propensity_table[self._X_flat]

        # Compute true APOs (after propensity table is built)
        self._true_apos = self._compute_true_apos()

        # Sample Y ~ Bernoulli(σ(μ(t, x)))
        mu = self._outcome_latent(self._T_multi, self._X_features)
        self._Y = rng.binomial(1, self._sigmoid(mu))

        # Treatment splits (on flat indices)
        train_t, val_t, test_t = self._split_treatments(rng)
        self._train_idx = np.where(np.isin(self._T_flat, train_t))[0]
        self._val_idx = np.where(np.isin(self._T_flat, val_t))[0]
        self._test_idx = np.where(np.isin(self._T_flat, test_t))[0]

    # ── Data access ────────────────────────────────────────────────────────

    @property
    def X(self) -> np.ndarray:
        """(n, d_x) covariate features."""
        return self._X_features

    @property
    def T(self) -> np.ndarray:
        """(n,) flat treatment indices in {0, ..., M-1}."""
        return self._T_flat

    @property
    def all_T_multi(self) -> np.ndarray:
        """(M, d_t) feature vectors for all M treatments."""
        return self._all_t_multi.astype(float)

    @property
    def T_multi(self) -> np.ndarray:
        """(n, d_t) multi-dimensional treatment features."""
        return self._T_multi

    @property
    def Y(self) -> np.ndarray:
        """(n,) discrete outcomes in {0, ..., n_outcomes-1}."""
        return self._Y

    @property
    def X_train(self) -> np.ndarray:
        return self._X_features[self._train_idx]

    @property
    def T_train(self) -> np.ndarray:
        return self._T_flat[self._train_idx]

    @property
    def T_multi_train(self) -> np.ndarray:
        return self._T_multi[self._train_idx]

    @property
    def Y_train(self) -> np.ndarray:
        return self._Y[self._train_idx]

    @property
    def X_val(self) -> np.ndarray:
        return self._X_features[self._val_idx]

    @property
    def T_val(self) -> np.ndarray:
        return self._T_flat[self._val_idx]

    @property
    def T_multi_val(self) -> np.ndarray:
        return self._T_multi[self._val_idx]

    @property
    def Y_val(self) -> np.ndarray:
        return self._Y[self._val_idx]

    @property
    def X_test(self) -> np.ndarray:
        return self._X_features[self._test_idx]

    @property
    def T_test(self) -> np.ndarray:
        return self._T_flat[self._test_idx]

    @property
    def T_multi_test(self) -> np.ndarray:
        return self._T_multi[self._test_idx]

    @property
    def Y_test(self) -> np.ndarray:
        return self._Y[self._test_idx]

    @property
    def propensity_scores_train(self) -> np.ndarray:
        return self._propensity_scores[self._train_idx]

    @property
    def propensity_scores_val(self) -> np.ndarray:
        return self._propensity_scores[self._val_idx]

    @property
    def propensity_scores_test(self) -> np.ndarray:
        return self._propensity_scores[self._test_idx]

    @property
    def n_train(self) -> int:
        return len(self._train_idx)

    @property
    def n_val(self) -> int:
        return len(self._val_idx)

    @property
    def n_test(self) -> int:
        return len(self._test_idx)

    # ── Ground-truth quantities ────────────────────────────────────────────

    @property
    def true_apos(self) -> np.ndarray:
        return self._true_apos

    @property
    def propensity_table(self) -> np.ndarray:
        return self._propensity_table

    @property
    def true_E_X(self) -> np.ndarray:
        return np.array([(v - 1) / 2.0 for v in self.FEATURE_SIZES])

    def x_moments(self, k: int) -> Dict[int, np.ndarray]:
        """Empirical moments of X_train up to order k.

        Returns:
            Dict mapping j -> (d_x,) array of E[X^j] for j = 1..k
        """
        return compute_x_moments(self.X_train, k)

    def naive_apo_estimates(self) -> np.ndarray:
        apos = np.full(self.M, np.nan)
        for t in range(self.M):
            mask = self._T_flat == t
            if mask.any():
                apos[t] = self._Y[mask].mean()
        return apos

    def confounding_bias(self) -> np.ndarray:
        return self.naive_apo_estimates() - self.true_apos

    # ── Tokenization ──────────────────────────────────────────────────────

    def make_token_sequences(self, X: np.ndarray, T_multi: np.ndarray) -> np.ndarray:
        """
        Encode (X, T) as (n, d_x + d_t) integer token sequences.

        Layout: [x_0+off_0, ..., x_{d_x-1}+off_{d_x-1},
                 t_0+toff_0, ..., t_{d_t-1}+toff_{d_t-1}]
        """
        x_offsets = np.array(self.x_vocab_offsets)
        t_offsets = np.array(self.t_vocab_offsets)
        x_tokens = X + x_offsets[np.newaxis, :]
        t_tokens = T_multi + t_offsets[np.newaxis, :]
        return np.concatenate([x_tokens, t_tokens], axis=1)

    def flat_t_to_multi(self, T_flat: np.ndarray) -> np.ndarray:
        """Convert (n,) flat treatment indices to (n, d_t) multi-dim."""
        return self._all_t_multi[T_flat]

    def decode_y_token(self, y_token: np.ndarray) -> np.ndarray:
        return np.asarray(y_token) - self.y_vocab_offset
"""
Projection comparison for AmazonReviewsProjections.

Trains ONE high-dim estimator (treat_dim="high", treatment T = full review text)
and THREE low-dim estimators (one per treatment_type: rating, sentiment, length).

For each treatment_type, the high-dim predictions are projected down to the
low-dim space via dataset.project_apos() and compared against:
  - the low-dim estimator trained directly on that treatment type
  - the ground-truth projected APOs

Usage:
    python projections.py dataset=amazon_reviews_projections estimator=outcome_imputation
    python projections.py dataset=amazon_reviews_projections estimator=ipw_crm
"""

import csv
import os

import hydra
import numpy as np
from omegaconf import DictConfig
from hydra.utils import instantiate
from scipy.stats import spearmanr

from src.evaluators.propensity_evaluator import PropensityEvaluator
from src.evaluators.sw_evaluator import SWEvaluator
from src.utils import seed_everything


TREATMENT_TYPES = ["rating", "sentiment", "length"]
fmt = lambda v: f"{v:.4f}" if v == v else "n/a" 


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fit_kwargs(estimator, dataset):
    """Build fit() kwargs from a dataset."""
    kwargs = {
        'T_multi':         dataset.T_multi_train,
        'val_T_multi':     dataset.T_multi_val,
        'all_T_multi':     dataset.all_T_multi,
        'feature_sizes':   dataset.feature_sizes,
        'treatment_sizes': dataset.treatment_sizes,
    }
    if hasattr(dataset, 'prompt_format'):
        kwargs['prompt_format'] = dataset.prompt_format
    if hasattr(estimator, 'propensity_model'):
        if hasattr(dataset, 'true_E_X'):
            kwargs['true_E_X'] = dataset.true_E_X
        if hasattr(dataset, 'x_moments'):
            k = getattr(estimator.propensity_model, 'balance_moments', 1)
            kwargs['balance_moment_targets'] = dataset.x_moments(k)
    if hasattr(estimator, 'w_model') and hasattr(dataset, 'x_moments'):
        k = getattr(estimator.w_model, 'balance_moments', 1)
        kwargs['balance_moment_targets'] = dataset.x_moments(k)
    return kwargs


def fit_estimator(estimator, dataset) -> tuple[np.ndarray, np.ndarray]:
    """Fit estimator, return (train_apos, val_apos) each shape (M,)."""
    estimator.fit(
        dataset.X_train, dataset.T_train, dataset.Y_train,
        dataset.X_val,   dataset.T_val,   dataset.Y_val,
        dataset.M,
        **_fit_kwargs(estimator, dataset),
    )
    return (
        estimator.predict_apos(dataset.X_train, dataset.M),
        estimator.predict_apos(dataset.X_val,   dataset.M),
    )


def eval_weights(estimator, dataset, X, T, T_multi) -> dict | None:
    if hasattr(estimator, 'propensity_model'):
        uses_table_path = (
            hasattr(estimator, 'train_propensity_table')
            and estimator.train_propensity_table is not None
        )
        if uses_table_path and not hasattr(dataset, 'propensity_table'):
            return None
        return PropensityEvaluator.evaluate(estimator, dataset, X, T, T_multi=T_multi)
    if hasattr(estimator, 'w_model'):
        return SWEvaluator.evaluate(estimator, dataset, X, T, T_multi)
    return None


def compute_metrics(estimated: np.ndarray, true: np.ndarray) -> dict:
    rel_errors = (estimated - true) / (true + 1e-12)
    pearson  = np.corrcoef(true, estimated)[0, 1] if len(true) > 1 else float('nan')
    spearman = spearmanr(true, estimated).statistic if len(true) > 1 else float('nan')
    return {
        'rel_mse':  float(np.mean(rel_errors ** 2)),
        'rel_mae':  float(np.mean(np.abs(rel_errors))),
        'pearson':  float(pearson),
        'spearman': float(spearman),
    }


def print_weight_metrics(label: str, wm_train: dict | None, wm_val: dict | None) -> None:
    if wm_train is None and wm_val is None:
        return
    wm_keys = ('avg_balance_score', 'avg_kl_divergence', 'marginal_kl_divergence')
    labels  = ('Avg balance score', 'Avg KL divergence', 'Marginal KL      ')
    print(f"  [{label} — weight quality]")
    print(f"    {'':22s}  {'train':>8s}  {'val':>8s}")
    for key, lbl in zip(wm_keys, labels):
        tr = fmt((wm_train or {}).get(key, float('nan')))
        va = fmt((wm_val   or {}).get(key, float('nan')))
        print(f"    {lbl}  {tr:>8s}  {va:>8s}")


def print_block(treatment_type, treatment_values, true_apos,
                train_high, val_high, train_low, val_low,
                wm_high, wm_low) -> tuple[dict, dict, dict, dict]:
    m_train_high = compute_metrics(train_high, true_apos)
    m_val_high   = compute_metrics(val_high,   true_apos)
    m_train_low  = compute_metrics(train_low,  true_apos)
    m_val_low    = compute_metrics(val_low,    true_apos)

    print(f"\n  treatment_type = {treatment_type}")
    print(f"  {'value':>22s}  {'true':>8s}  "
          f"{'hi_tr':>8s}  {'hi_val':>8s}  {'lo_tr':>8s}  {'lo_val':>8s}")
    print(f"  {'-'*74}")
    for v, gt, htr, hva, ltr, lva in zip(
            treatment_values, true_apos, train_high, val_high, train_low, val_low):
        print(f"  {v:>22s}  {gt:8.4f}  {htr:8.4f}  {hva:8.4f}  {ltr:8.4f}  {lva:8.4f}")

    print(f"\n  {'metric':<12}  {'hi_train':>10}  {'hi_val':>8}  "
          f"{'lo_train':>10}  {'lo_val':>8}")
    print(f"  {'-'*54}")
    for key in ('rel_mse', 'rel_mae', 'pearson', 'spearman'):
        print(f"  {key:<12}  {m_train_high[key]:10.4f}  {m_val_high[key]:8.4f}  "
              f"{m_train_low[key]:10.4f}  {m_val_low[key]:8.4f}")

    if wm_high['train'] is not None or wm_low['train'] is not None:
        print()
        print_weight_metrics(f"high, tt={treatment_type}", wm_high['train'], wm_high['val'])
        print_weight_metrics(f"low,  tt={treatment_type}", wm_low['train'],  wm_low['val'])

    return m_train_high, m_val_high, m_train_low, m_val_low


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    seed = cfg.get('seed', 42)
    seed_everything(seed)

    print("=" * 70)
    print("APO Projection Comparison  (high-dim vs low-dim, all treatment types)")
    print("=" * 70)
    print(f"  estimator : {cfg.estimator._target_.split('.')[-1]}")

    # ── Build datasets ────────────────────────────────────────────────────────
    ds_high = {
        tt: instantiate(cfg.dataset, treatment_type=tt, treat_dim="high")
        for tt in TREATMENT_TYPES
    }
    ds_low = {
        tt: instantiate(cfg.dataset, treatment_type=tt, treat_dim="low")
        for tt in TREATMENT_TYPES
    }

    for tt in TREATMENT_TYPES:
        assert ds_high[tt].treatment_values == ds_low[tt].treatment_values
        assert np.allclose(ds_high[tt].true_apos, ds_low[tt].true_apos)

    # ── Train ONE high-dim estimator ──────────────────────────────────────────
    # All ds_high[tt] share the same T_text (treat_dim="high" always uses "t").
    print("\n" + "-" * 70)
    print("Training high-dim estimator on full review text...")
    estimator_high = instantiate(cfg.estimator)
    raw_train_high, raw_val_high = fit_estimator(estimator_high, ds_high["rating"])
    print(f"  M={ds_high['rating'].M}, raw APO shape: {raw_train_high.shape}")

    # ── Train one low-dim estimator per treatment_type ────────────────────────
    all_metrics = {}
    for tt in TREATMENT_TYPES:
        print("\n" + "-" * 70)
        print(f"Training low-dim estimator  (treatment_type={tt})...")

        estimator_low = instantiate(cfg.estimator)
        train_low, val_low = fit_estimator(estimator_low, ds_low[tt])

        # Project high-dim APOs for this treatment_type.
        ds = ds_high[tt]
        train_high = ds.project_apos(raw_train_high, ds._T_low_text)
        val_high   = ds.project_apos(raw_val_high,   ds._T_low_text)

        # Weight quality for both splits.
        wm_high, wm_low = {}, {}
        for split, X, T, T_multi in [
            ('train', ds_high[tt].X_train, ds_high[tt].T_train, ds_high[tt].T_multi_train),
            ('val',   ds_high[tt].X_val,   ds_high[tt].T_val,   ds_high[tt].T_multi_val),
        ]:
            wm_high[split] = eval_weights(estimator_high, ds_high[tt], X, T, T_multi)

        for split, X, T, T_multi in [
            ('train', ds_low[tt].X_train, ds_low[tt].T_train, ds_low[tt].T_multi_train),
            ('val',   ds_low[tt].X_val,   ds_low[tt].T_val,   ds_low[tt].T_multi_val),
        ]:
            wm_low[split] = eval_weights(estimator_low, ds_low[tt], X, T, T_multi)

        print(f"\n  Results for treatment_type={tt}:")
        m_tr_hi, m_va_hi, m_tr_lo, m_va_lo = print_block(
            tt, ds.treatment_values, ds.true_apos,
            train_high, val_high, train_low, val_low,
            wm_high, wm_low,
        )
        all_metrics[tt] = {
            'train_high': m_tr_hi, 'val_high': m_va_hi,
            'train_low':  m_tr_lo, 'val_low':  m_va_lo,
            'wm_high': wm_high,   'wm_low':   wm_low,
        }

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Summary")
    mkeys = ('rel_mse', 'rel_mae', 'pearson', 'spearman')
    header = (f"  {'treatment_type':<12}  {'dim':<6}  {'split':<6}"
              + "".join(f"  {k:>10}" for k in mkeys))
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for tt in TREATMENT_TYPES:
        for split in ('train', 'val'):
            for dim in ('high', 'low'):
                vals = "".join(
                    f"  {all_metrics[tt][f'{split}_{dim}'][k]:>10.4f}" for k in mkeys
                )
                print(f"  {tt:<12}  {dim:<6}  {split:<6}{vals}")
        print()

    # ── Optional CSV output ───────────────────────────────────────────────────
    output_csv = cfg.get('output_csv', None)
    if output_csv:
        estimator_name = estimator_high.__class__.__name__
        wm_keys = ('avg_balance_score', 'avg_kl_divergence', 'marginal_kl_divergence')
        rows = []
        for tt in TREATMENT_TYPES:
            ds = ds_high[tt]
            row = {
                'seed':            seed,
                'experiment_name': cfg.get('experiment_name', ''),
                'estimator':       estimator_name,
                'dataset':         ds.__class__.__name__,
                'dataset_n':       ds.n,
                'category':        getattr(ds, 'category', ''),
                'num_bins':        getattr(ds, 'num_bins', ''),
                'treatment_type':  tt,
            }
            for split in ('train', 'val'):
                for dim in ('high', 'low'):
                    for k, v in all_metrics[tt][f'{split}_{dim}'].items():
                        row[f'{split}_{dim}_{k}'] = v
                    wm = all_metrics[tt][f'wm_{dim}'].get(split) or {}
                    for k in wm_keys:
                        row[f'{split}_{dim}_{k}'] = wm.get(k, '')
            rows.append(row)

        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        write_header = not os.path.exists(output_csv)
        with open(output_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults appended to {output_csv}")


if __name__ == "__main__":
    main()

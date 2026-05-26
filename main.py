import csv
import os

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from src.evaluators.apo_evaluator import APOEvaluator
from src.evaluators.propensity_evaluator import PropensityEvaluator
from src.evaluators.sw_evaluator import SWEvaluator
from src.utils import seed_everything


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Set random seed for reproducibility
    seed = cfg.get('seed', 42)
    seed_everything(seed)

    print("=" * 70)
    print("APO Estimation Example with Hydra")
    print("=" * 70)
    print(f"Random seed: {seed}")

    # 1. Instantiate dataset from config
    print("\n1. Creating dataset from config...")
    dataset = instantiate(cfg.dataset)
    print(f"   Total samples: {dataset.n}")
    print(f"   Splits: train={dataset.n_train}, val={dataset.n_val}, test={dataset.n_test}")
    print(f"   Dimensions: {dataset.M} treatments, feature_sizes={dataset.feature_sizes}")
    print(f"   True APOs: {dataset.true_apos}")

    # 2. Instantiate estimator from config
    print("\n2. Creating estimator from config...")
    estimator = instantiate(cfg.estimator)
    print(f"   Estimator: {estimator.__class__.__name__}")

    # 3. Evaluate
    print("\n3. Training and evaluating estimator...")
    results = APOEvaluator.evaluate(estimator, dataset)
    print(f"   Train APOs: {results['train_estimated_apos'][:10]}")
    print(f"   Val APOs:   {results['val_estimated_apos'][:10]}")
    print(f"\n   Train Metrics:")
    print(f"     Rel MSE:      {results['train_rel_mse']:.4f}")
    print(f"     Rel MAE:      {results['train_rel_mae']:.4f}")
    print(f"     Pearson r:    {results['train_pearson']:.4f}")
    print(f"     Spearman r:   {results['train_spearman']:.4f}")
    print(f"\n   Val Metrics:")
    print(f"     Rel MSE:      {results['val_rel_mse']:.4f}")
    print(f"     Rel MAE:      {results['val_rel_mae']:.4f}")
    print(f"     Pearson r:    {results['val_pearson']:.4f}")
    print(f"     Spearman r:   {results['val_spearman']:.4f}")

    # 4. Weight quality evaluation
    prop_train, prop_val = None, None
    fmt = lambda v: f"{v:.4f}" if v == v else "n/a"  # noqa: E731
    if hasattr(estimator, 'propensity_model'):
        print("\n4. Propensity evaluation...")
        for split, X, T, T_multi in [
            ('train', dataset.X_train, dataset.T_train, getattr(dataset, 'T_multi_train', None)),
            ('val',   dataset.X_val,   dataset.T_val,   getattr(dataset, 'T_multi_val',   None)),
        ]:
            prop = PropensityEvaluator.evaluate(estimator, dataset, X, T, T_multi=T_multi)
            print(f"\n   {split.capitalize()}:")
            print(f"     Avg balance score:    {fmt(prop['avg_balance_score'])}")
            print(f"     Avg KL divergence:    {fmt(prop['avg_kl_divergence'])}")
            print(f"     Marginal KL:          {fmt(prop['marginal_kl_divergence'])}")
            if split == 'train':
                prop_train = prop
            else:
                prop_val = prop
    elif hasattr(estimator, 'w_model'):
        print("\n4. Importance weight evaluation...")
        for split, X, T, T_multi in [
            ('train', dataset.X_train, dataset.T_train, getattr(dataset, 'T_multi_train', None)),
            ('val',   dataset.X_val,   dataset.T_val,   getattr(dataset, 'T_multi_val',   None)),
        ]:
            iw = SWEvaluator.evaluate(estimator, dataset, X, T, T_multi)
            print(f"\n   {split.capitalize()}:")
            print(f"     Avg balance score:    {fmt(iw['avg_balance_score'])}")
            if split == 'train':
                prop_train = iw
            else:
                prop_val = iw

    print("\n" + "=" * 70)
    print(f"Model trained on {dataset.n_train} samples")
    print(f"APOs estimated using {dataset.n_val} validation samples")
    print("=" * 70)

    # 5. Write results to CSV if requested
    output_csv = cfg.get('output_csv', None)
    if output_csv:
        apo_metric_keys = ('rel_mse', 'rel_mae', 'pearson', 'spearman')
        wm_keys = ('avg_balance_score', 'avg_kl_divergence', 'marginal_kl_divergence')
        apo_metrics = {
            split: {k: results[f'{split}_{k}'] for k in apo_metric_keys}
            for split in ('train', 'val')
        }
        wm = {'train': prop_train, 'val': prop_val}

        row = {
            'seed':              seed,
            'experiment_name':   cfg.get('experiment_name', ''),
            'estimator':         estimator.__class__.__name__,
            'dataset':           dataset.__class__.__name__,
            'dataset_n':         dataset.n,
            'M':                 getattr(dataset, 'M', ''),
            'num_bins':          getattr(dataset, 'num_bins', ''),
            'category':          getattr(dataset, 'category', ''),
            'confound_strength': getattr(dataset, 'confound_strength', ''),
            'n_x_values':        getattr(dataset, 'n_x_values', ''),
        }
        for split in ('train', 'val'):
            for k, v in apo_metrics[split].items():
                row[f'{split}_{k}'] = v
            for k in wm_keys:
                row[f'{split}_{k}'] = (wm[split] or {}).get(k, '')

        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        write_header = not os.path.exists(output_csv)
        with open(output_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(f"\nResults appended to {output_csv}")


if __name__ == "__main__":
    main()

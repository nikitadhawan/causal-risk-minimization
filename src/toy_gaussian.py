"""
Gaussian Linear APO Estimation.
Disjoint treatment ranges: train on middle, validate on tails (unseen treatments).
"""

import torch
import torch.nn as nn
from torch.optim import Adam

# ── DGP ──────────────────────────────────────────────────────────────────────

def generate_data(n, mu_x=0., sig_x=1., alpha=0., beta=1., sig_tx=1.,
                  g0=1., gT=2., gX=3., sig_eps=1):
    X = mu_x + sig_x * torch.randn(n)
    T = alpha + beta * X + sig_tx * torch.randn(n)
    Y = g0 + gT * T + gX * X + sig_eps * torch.randn(n)
    p = dict(mu_x=mu_x, sig_x=sig_x, alpha=alpha, beta=beta,
             sig_tx=sig_tx, g0=g0, gT=gT, gX=gX)
    return X, T, Y, p

def true_apo(t, p):
    return p['g0'] + p['gT'] * t + p['gX'] * p['mu_x']

def split_by_treatment(X, T, Y, train_frac=0.6, seed=42):
    """Randomly split data into train/val sets."""
    rng = torch.Generator().manual_seed(seed)
    n = len(T)
    perm = torch.randperm(n, generator=rng)
    n_train = int(n * train_frac)
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    return (X[train_idx], T[train_idx], Y[train_idx],
            X[val_idx], T[val_idx], Y[val_idx])

# ── Propensity model ─────────────────────────────────────────────────────────

class PropensityMLP(nn.Module):
    def __init__(self, hidden=64, n_layers=3):
        super().__init__()
        layers = [nn.Linear(1, hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        self.net = nn.Sequential(*layers)
        self.mu_head = nn.Linear(hidden, 1)
        self.logs_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.net(x.unsqueeze(1))
        return self.mu_head(h).squeeze(1), self.logs_head(h).squeeze(1)

    def log_prob(self, x, t):
        mu, log_s = self(x)
        return -0.5 * ((t - mu) / log_s.exp())**2 - log_s - 0.9189

    def weights(self, X, T, clip=3.0, differentiable=True):
        ctx = torch.enable_grad() if differentiable else torch.no_grad()
        with ctx:
            mu, log_s = self(X)
            sig = log_s.exp()
            mu_marg = mu.mean()
            sig_marg2 = mu.var() + (sig**2).mean()
            log_marg = -0.5 * (T - mu_marg)**2 / sig_marg2 - 0.5 * sig_marg2.log()
            log_cond = -0.5 * ((T - mu) / sig)**2 - log_s
            w = (log_marg - log_cond).clamp(max=clip).exp()
        return w

def analytical_weights(T, X, alpha, beta, sig_tx, X_all):
    """Oracle weights using true parameters. X_all for marginal estimation."""
    mu_t = alpha + beta * X_all.mean()
    sig_t2 = beta**2 * X_all.var() + sig_tx**2
    log_w = (-0.5 * (T - mu_t)**2 / sig_t2
             + 0.5 * (T - alpha - beta * X)**2 / sig_tx**2
             + 0.5 * torch.log(sig_tx**2 / sig_t2))
    return torch.exp(log_w)

# ── Penalties ────────────────────────────────────────────────────────────────

def penalties(X, w, mu_x):
    bal = ((w * X - mu_x) ** 2).mean()
    norm = ((w - 1.0) ** 2).mean()
    return bal, norm

# ── Training ─────────────────────────────────────────────────────────────────

def train_propensity(X, T, mu_x, lam_bal=0., lam_norm=0.,
                     hidden=64, n_layers=3, lr=1e-3, steps=2000):
    model = PropensityMLP(hidden, n_layers)
    opt = Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        nll = -model.log_prob(X, T).mean()
        if lam_bal > 0 or lam_norm > 0:
            w = model.weights(X, T, differentiable=True)
            bal, norm = penalties(X, w, mu_x)
        else:
            bal, norm = torch.tensor(0.), torch.tensor(0.)
        loss = nll + lam_bal * bal + lam_norm * norm
        opt.zero_grad(); loss.backward(); opt.step()
    return model

def train_apo(T, Y, w, lr=0.01, steps=2000):
    model = nn.Linear(1, 1)
    opt = Adam(model.parameters(), lr=lr)
    targets = (w * Y).detach()
    for _ in range(steps):
        pred = model(T.unsqueeze(1)).squeeze()
        loss = ((pred - targets)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model

def train_oi(T, X, Y, lr=0.01, steps=2000):
    model = nn.Linear(2, 1)
    opt = Adam(model.parameters(), lr=lr)
    inp = torch.stack([T, X], dim=1)
    for _ in range(steps):
        pred = model(inp).squeeze()
        loss = ((pred - Y)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model

# ── Evaluation helpers ───────────────────────────────────────────────────────

def eval_apo(model, t_grid):
    with torch.no_grad():
        return torch.tensor([model(t.view(1, 1)).item() for t in t_grid])

def eval_oi(model, X, t_grid):
    vals = []
    with torch.no_grad():
        for t in t_grid:
            inp = torch.stack([t.expand_as(X), X], dim=1)
            vals.append(model(inp).squeeze().mean().item())
    return torch.tensor(vals)

def balance_error(X, w, mu_x):
    """Weighted mean of X minus mu_x: E_w[X] - mu_x."""
    return ((w * X).sum() / w.sum() - mu_x).item()

def norm_check(w):
    """Mean weight: should be 1 for well-calibrated propensity."""
    return w.mean().item()

def mse(pred, gt):
    p = pred.detach() if isinstance(pred, torch.Tensor) else torch.tensor(pred)
    g = gt.detach() if isinstance(gt, torch.Tensor) else torch.tensor(gt)
    return ((p - g)**2).mean().item()

def mae(pred, gt):
    p = pred.detach() if isinstance(pred, torch.Tensor) else torch.tensor(pred)
    g = gt.detach() if isinstance(gt, torch.Tensor) else torch.tensor(gt)
    return (p - g).abs().mean().item()

# ── Sample-size ablation ──────────────────────────────────────────────────────

def run_n_ablation(
    sample_sizes=(64, 128, 256, 512, 1024),
    n_trials: int = 3,
    lam_bal: float = 10.,
    lam_norm: float = 10.,
    output_path: str = 'results/gaussian_n_ablation.png',
):
    """
    Sweep over sample sizes, tracking train/val relative MSE, balance error,
    and normalization error for OI, IPW, and Bal+Norm-IPW.
    """
    import numpy as np

    estimators = ['OI', 'IPW', 'Bal+Norm']
    ipw_names  = ['IPW', 'Bal+Norm']
    N = len(sample_sizes)

    # shape: (N, n_trials)
    rel_mse  = {e: {'train': np.zeros((N, n_trials)), 'val': np.zeros((N, n_trials))}
                for e in estimators}
    rel_mae  = {e: {'train': np.zeros((N, n_trials)), 'val': np.zeros((N, n_trials))}
                for e in estimators}
    bal_err  = {e: {'train': np.zeros((N, n_trials)), 'val': np.zeros((N, n_trials))}
                for e in ipw_names}
    norm_err = {e: {'train': np.zeros((N, n_trials)), 'val': np.zeros((N, n_trials))}
                for e in ipw_names}

    for i, n in enumerate(sample_sizes):
        for trial in range(n_trials):
            seed = 42 + trial
            torch.manual_seed(seed)
            X, T, Y, p = generate_data(n)
            Xtr, Ttr, Ytr, Xv, Tv, Yv = split_by_treatment(
                X, T, Y, train_frac=0.6, seed=seed)

            t_train_grid = torch.linspace(Ttr.min(), Ttr.max(), 40)
            t_val_grid   = torch.linspace(Tv.min(),  Tv.max(),  40)
            gt_train = torch.tensor([true_apo(t, p) for t in t_train_grid])
            gt_val   = torch.tensor([true_apo(t, p) for t in t_val_grid])
            gt_train_var = gt_train.var().item() + 1e-12
            gt_val_var   = gt_val.var().item()   + 1e-12
            gt_train_std = gt_train.std().item() + 1e-12
            gt_val_std   = gt_val.std().item()   + 1e-12

            # OI
            oi = train_oi(Ttr, Xtr, Ytr)
            pred_tr = eval_oi(oi, Xtr, t_train_grid)
            pred_v  = eval_oi(oi, Xv,  t_val_grid)
            rel_mse['OI']['train'][i, trial] = mse(pred_tr, gt_train) / gt_train_var
            rel_mse['OI']['val'  ][i, trial] = mse(pred_v,  gt_val)   / gt_val_var
            rel_mae['OI']['train'][i, trial] = mae(pred_tr, gt_train) / gt_train_std
            rel_mae['OI']['val'  ][i, trial] = mae(pred_v,  gt_val)   / gt_val_std

            # IPW estimators
            configs = [
                ('IPW',      0.,      0.),
                ('Bal+Norm', lam_bal, lam_norm),
            ]
            for name, lb, ln in configs:
                prop = train_propensity(Xtr, Ttr, p['mu_x'], lam_bal=lb, lam_norm=ln)
                w_tr = prop.weights(Xtr, Ttr, differentiable=False)
                w_v  = prop.weights(Xv,  Tv,  differentiable=False)
                apo  = train_apo(Ttr, Ytr, w_tr)
                pred_tr = eval_apo(apo, t_train_grid)
                pred_v  = eval_apo(apo, t_val_grid)
                rel_mse[name]['train'][i, trial] = mse(pred_tr, gt_train) / gt_train_var
                rel_mse[name]['val'  ][i, trial] = mse(pred_v,  gt_val)   / gt_val_var
                rel_mae[name]['train'][i, trial] = mae(pred_tr, gt_train) / gt_train_std
                rel_mae[name]['val'  ][i, trial] = mae(pred_v,  gt_val)   / gt_val_std

                # balance: |E_w[X] - mu_x|
                bal_err[name]['train'][i, trial] = abs(balance_error(Xtr, w_tr, p['mu_x']))
                bal_err[name]['val'  ][i, trial] = abs(balance_error(Xv,  w_v,  p['mu_x']))

                # norm: |E[w] - 1|
                norm_err[name]['train'][i, trial] = abs(norm_check(w_tr) - 1.)
                norm_err[name]['val'  ][i, trial] = abs(norm_check(w_v)  - 1.)

            print(f"  n={n:7d}  trial={trial}  "
                  f"OI val rel_mae={rel_mae['OI']['val'][i,trial]:.4f}  "
                  f"IPW val rel_mae={rel_mae['IPW']['val'][i,trial]:.4f}  "
                  f"BN val rel_mae={rel_mae['Bal+Norm']['val'][i,trial]:.4f}")

    csv_path = output_path.replace('.png', '.csv')
    save_n_ablation_csv(sample_sizes, rel_mse, rel_mae, bal_err, norm_err,
                        estimators, ipw_names, csv_path)
    return rel_mse, rel_mae, bal_err, norm_err


def save_n_ablation_csv(sample_sizes, rel_mse, rel_mae, bal_err, norm_err,
                        estimators, ipw_names, csv_path):
    import csv, math
    rows = []
    n_trials = rel_mse[estimators[0]]['train'].shape[1]
    for i, n in enumerate(sample_sizes):
        for trial in range(n_trials):
            for e in estimators:
                for split in ('train', 'val'):
                    rows.append({
                        'n':        n,
                        'trial':    trial,
                        'estimator': e,
                        'split':    split,
                        'rel_mse':  rel_mse[e][split][i, trial],
                        'rel_mae':  rel_mae[e][split][i, trial],
                        'bal_err':  bal_err[e][split][i, trial] if e in bal_err else math.nan,
                        'norm_err': norm_err[e][split][i, trial] if e in norm_err else math.nan,
                    })
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['n', 'trial', 'estimator', 'split',
                           'rel_mse', 'rel_mae', 'bal_err', 'norm_err'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to: {csv_path}")


if __name__ == '__main__':
    run_n_ablation(
        sample_sizes=(64, 128, 256, 512, 1024),
        n_trials=20,
    )
# CRM: Causal Risk Minimization

This codebase implements the experiments and reproduces the results of [Causal Risk Minimization
for High-Dimensional Treatments](https://arxiv.org/abs/2605.27281).

CRM performs Average Potential Outcome (APO) estimation for high-dimensional treatments, e.g. text.

## Setup

Requires Python 3.11+. Clone the repo and install:

```bash
git clone https://github.com/nikitadhawan/causal-risk-minimization
cd causal-risk-minimization
pip install -e .
```

## Experiments

The two main entry points are:

- `main.py` runs a single experiment with a chosen APO estimator and dataset.
- `projections.py` projects Amazon Reviews APOs onto lower-dimensional attributes.

## Example usage

### Continuous Gaussian

```bash
python src/toy_gaussian.py
```

### Synthetic Discrete

```bash
python main.py \
    estimator=sw_crm \
    model/apo@estimator.apo_model=transformer \
    model/sw@estimator.w_model=transformer \
    dataset=toy_text \
    estimator.w_model.balance_weight=1 \
    estimator.w_model.norm_weight=10 \
    estimator.w_model.balance_moments=1 \
    estimator.apo_model.epochs=1500 \
    output_csv=results/synthetic_results.csv
```

### Amazon Reviews (Text)

```bash
python main.py \
    estimator=sw_crm \
    model/apo@estimator.apo_model=hf \
    model/sw@estimator.w_model=hf \
    dataset=amazon_reviews \
    estimator.w_model.model_name=google/gemma-3-270m \
    estimator.w_model.balance_weight=0.1 \
    estimator.w_model.norm_weight=1 \
    estimator.w_model.epochs=10 \
    estimator.apo_model.model_name=google/gemma-3-270m \
    estimator.apo_model.lr=1e-5 \
    estimator.apo_model.epochs=5 \
    output_csv=results/amazon_results.csv
```

### Amazon Reviews Projections

```bash
python projections.py \
    estimator=sw_crm \
    model/apo@estimator.apo_model=hf \
    model/sw@estimator.w_model=hf \
    dataset=amazon_reviews_projections \
    estimator.w_model.model_name=google/gemma-3-270m \
    estimator.w_model.balance_weight=2 \
    estimator.w_model.norm_weight=10 \
    estimator.w_model.lr=1e-5 \
    estimator.w_model.epochs=10 \
    estimator.apo_model.model_name=google/gemma-3-270m \
    estimator.apo_model.lr=1e-5 \
    estimator.apo_model.epochs=10 \
    seed=2 \
    output_csv=results/amazon_projections_results.csv
```

## Project Structure

```
.
├── src/
│   ├── datasets/
│   │   ├── toy_text.py                    # Synthetic discrete dataset
│   │   ├── amazon_reviews.py              # Amazon Electronics reviews dataset
│   │   ├── amazon_reviews_projections.py  # Amazon Reviews with attribute projections
│   │   ├── generate_amazon_reviews.py     # Dataset generation script
│   │   ├── add_treatment_columns.py       # Preprocessing utility
│   │   └── data/amazon/                   # Preprocessed Amazon data files
│   ├── estimators/
│   │   ├── base.py                        # Abstract base class
│   │   ├── outcome_imputation.py          # Outcome imputation (OI)
│   │   ├── oi_crm.py                      # OI-CRM estimator
│   │   ├── ipw_crm.py                     # IPW-CRM estimator
│   │   └── sw_crm.py                      # SW-CRM estimator
│   ├── evaluators/
│   │   ├── apo_evaluator.py               # APO evaluation metrics
│   │   ├── propensity_evaluator.py        # Propensity model evaluation
│   │   └── sw_evaluator.py                # SW model evaluation
│   ├── models/
│   │   ├── apo_models.py                  # APO model interface
│   │   ├── outcome_models.py              # Outcome model interface
│   │   ├── propensity_models.py           # Propensity model interface
│   │   ├── sw_models.py                   # SW model interface
│   │   ├── linear.py                      # Linear model
│   │   ├── mlp.py                         # MLP model
│   │   ├── transformer.py                 # Shared transformer backbone
│   │   ├── transformer_apo.py             # Transformer APO model
│   │   ├── transformer_outcome.py         # Transformer outcome model
│   │   ├── transformer_propensity.py      # Transformer propensity model
│   │   ├── transformer_sw.py              # Transformer SW model
│   │   ├── hf_apo.py                      # HuggingFace APO wrapper
│   │   ├── hf_outcome.py                  # HuggingFace outcome wrapper
│   │   ├── hf_propensity.py               # HuggingFace propensity wrapper
│   │   ├── hf_sw.py                       # HuggingFace SW wrapper
│   │   └── prompt_format.py               # Prompt formatting for HF models
│   ├── toy_gaussian.py                    # Continuous Gaussian experiment
│   └── utils.py                           # Shared utilities
├── conf/
│   ├── config.yaml                        # Main Hydra config
│   ├── dataset/
│   │   ├── toy_text.yaml
│   │   ├── amazon_reviews.yaml
│   │   └── amazon_reviews_projections.yaml
│   ├── estimator/
│   │   ├── outcome_imputation.yaml
│   │   ├── oi_crm.yaml
│   │   ├── ipw_crm.yaml
│   │   └── sw_crm.yaml
│   └── model/
│       ├── apo/        # {hf, linear, mlp, transformer}
│       ├── outcome/    # {hf, linear, mlp, transformer}
│       ├── propensity/ # {empirical, hf, linear, mlp, oracle, transformer}
│       └── sw/         # {hf, linear, mlp, transformer}
├── main.py                                # Main experiment entry point
├── projections.py                         # Amazon Reviews Projections entry point
└── pyproject.toml
```

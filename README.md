# Tunable Split Conformal Prediction experiments

This repository contains a reproducible implementation of Tunable Split Conformal Prediction (TsCP) and the paper experiments. The new implementation follows the revised coverage theory:

\[
1-\widehat{\mathbb E}[\alpha_\delta(X)]-\widehat{\mathbb E}[\Delta_{\delta,n}],
\]

not the old proxy `1 - mean(alpha_delta)`.

The original scripts remain at the repository root and in `experiments/` for comparison. New experiments should use `src/tscp`, YAML configs, and the scripts below.

## Installation

Python 3.10 or newer is required; Python 3.12 is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Linux or macOS, activate with `source .venv/bin/activate`. Alternatively:

```bash
python -m pip install -r requirements.txt
```

## Calibration and budget conventions

`total_calibration_size` always means

\[
N_{\mathrm{cal}}=2n, \qquad |D_1|=|D_2|=n.
\]

The budget is a user-specified function `S(x)` fixed independently of calibration and test labels. Four theoretically compatible parameterizations are implemented:

- `constant`: `S(x) = C`;
- `linear`: interpolate between `minimum` and `maximum` using normalized uncertainty `u(x)`;
- `quadratic`: interpolate using `u(x)^2`;
- `exponential`: interpolate using `(exp(beta*u(x))-1)/(exp(beta)-1)`.

Regression uncertainty comes from the training-fitted scale model. Classification uncertainty uses predictive entropy or margin. Classification budgets are rounded upward to an integer and all budgets are clamped to configured bounds.

The code validates `S(x) - delta >= 0`. The soft-size theorem additionally requires `delta > L`. In finite-label classification, the local jump bound is commonly `L=1`, so theory-oriented configs use `delta > 1` rather than exactly `1`.

## Downloading datasets

Synthetic datasets need no download. Cache real datasets before a full run:

```bash
python scripts/download_data.py california_housing superconductivity covertype mnist
```

MNIST, Covertype, and Superconductivity require network access on their first run.

## Running experiments

Run one experiment:

```bash
python scripts/run_experiment.py --config configs/experiments/self_validation_regression.yaml
```

Run a small synthetic smoke version first:

```bash
python scripts/run_experiment.py --config configs/experiments/self_validation_regression.yaml --smoke
```

Override YAML values without editing a config:

```bash
python scripts/run_experiment.py --config configs/experiments/delta_regression.yaml --set method.delta=0.2 --set data.total_calibration_size=4000
```

Run the paper suites:

```bash
python scripts/run_suite.py --suite configs/suites/main_paper.yaml
python scripts/run_suite.py --suite configs/suites/appendix.yaml
```

Smoke-test a complete suite:

```bash
python scripts/run_suite.py --suite configs/suites/main_paper.yaml --smoke
```

## Experiment configs

| Config | Output |
|---|---|
| `self_validation_*.yaml` | Corrected coverage vs. test count and size vs. total calibration size |
| `delta_*.yaml` | Coverage and average size as functions of slack `delta` |
| `compare_ecp_*.yaml` | Coverage-size comparison of TsCP variants and eCP |
| `hard_constraint*.yaml` | Hard-constraint satisfaction tables |
| `runtime.yaml` | A 2x3 figure for sCP, TsCP, eCP, and eCP-TPSS |
| `budget_ablation.yaml` | Constant, linear, quadratic, and exponential budgets |
| `model_ablation_*.yaml` | Coverage and set size vs. total calibration size for multiple models |
| `loo_validation_*.yaml` | LOO errors for both `alpha_hat` and the revised `delta_hat` |

## Outputs

Every invocation creates a new run directory:

```text
outputs/<experiment>/<timestamp>/
├── config.resolved.yaml
├── environment.json
├── metrics.csv
├── figure.pdf
└── figure.png
```

Hard-constraint runs additionally write CSV and LaTeX tables. Regenerate a figure without fitting models again:

```bash
python scripts/make_figures.py --run-dir outputs/<experiment>/<timestamp>
```

## Corrected coverage estimator

For each observation in `D1`, the implementation removes that observation, recomputes its adaptive level, and evaluates its pseudo-error against the independent `D2` quantile. Saved metrics include:

- `alpha_hat`: LOO estimate of `E[alpha_delta(X)]`;
- `delta_hat`: LOO estimate of `E[Delta_delta,n]`;
- `old_proxy`: `1 - alpha_hat`, retained only for comparison;
- `corrected_bound`: `1 - alpha_hat - delta_hat`, used for theoretical curves.

The implementation is in `src/tscp/theory/coverage.py`. The old proxy must not be labeled as the revised theorem's coverage bound.

## Repository structure

```text
configs/              YAML experiment and suite definitions
scripts/              Stable command-line entry points
src/tscp/             Shared methods, theory, experiment, and plotting code
tests/unit/            Mathematical and API unit tests
tests/smoke/           Small end-to-end synthetic tests
outputs/               Generated metrics and figures; ignored by version control
experiments/           Legacy experiment scripts retained for comparison
submission/            Legacy submission snapshot; not a source directory
```

## Tests

```bash
pytest -q
```

Tests cover conformal quantile indexing, budget validity, corrected LOO coverage construction, and an end-to-end synthetic self-validation run.

## Reproducibility notes

- Dataset splitting, calibration sampling, and model fitting are seeded.
- Raw metrics are saved before plotting.
- Figures aggregate over configured seeds.
- Model fitting is excluded from runtime measurements; runtime covers the conformal inference layer.
- Regenerate submission artifacts from the new source after results are finalized rather than editing the legacy copies.

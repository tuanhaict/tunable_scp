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
| `self_validation_*.yaml` | Empirical coverage vs. the independent-reference target `1-E[alpha]-E[Delta]`, and size vs. total calibration size |
| `delta_*.yaml` | Coverage and average size as functions of slack `delta` |
| `compare_ecp_*.yaml` | Coverage-size comparison of TsCP variants and eCP |
| `hard_constraint*.yaml` | Hard-constraint satisfaction tables |
| `runtime.yaml` | A 2x3 figure for sCP, TsCP, eCP, and eCP-TPSS |
| `budget_ablation.yaml` | Constant, linear, quadratic, and exponential budgets |
| `model_ablation_*.yaml` | Coverage and set size vs. total calibration size for multiple models |
| `loo_validation_*.yaml` | Histograms of independent-test coverage and the corrected LOO estimate `1-alpha_hat_LOO-delta_hat_LOO` |
| `loo_compare_ecp_*.yaml` | Variance and absolute-error comparison: truncated eCP uses `alpha_hat_LOO`, TsCP uses `alpha_hat_LOO + delta_hat_LOO` |

Run the revised LOO estimator comparison with:

```bash
python scripts/run_experiment.py --config configs/experiments/loo_compare_ecp_regression.yaml
python scripts/run_experiment.py --config configs/experiments/loo_compare_ecp_classification.yaml
```

For each outer seed and calibration size, the comparison performs multiple
calibration trials. The truncated-eCP estimator is `alpha_hat_loo`; the corrected
TsCP estimator is `alpha_hat_loo + delta_hat_loo`. Every trial resamples the
calibration subset. A separate, independent `reference_trials` stream resamples
both calibration and exactly one random test point per draw to estimate the
fixed target expectation. Errors are then computed as
`abs(loo_estimate - reference_target)`. Lines and bands report the mean and one
standard deviation over outer seeds. The CSV also records the reference alpha,
delta, target, and Monte Carlo standard error.

LOO comparison runs additionally write a compact `loo_compare_points.csv` with
one row per dataset/calibration size and only the four plotted mean values:
eCP/TsCP variance and eCP/TsCP absolute error. The more detailed per-outer-seed
statistics remain available in `loo_compare_points_by_seed.csv`.
The third plot compares each LOO coverage estimate with independently estimated
empirical coverage. Its report values are saved in `loo_coverage_points.csv`.

Run the regression or classification LOO histogram with:

```bash
python scripts/run_experiment.py --config configs/experiments/loo_validation_regression.yaml
python scripts/run_experiment.py --config configs/experiments/loo_validation_classification.yaml
```

Each Monte Carlo trial redraws `D1` and `D2`. The blue value is
`test_corrected = 1 - test_alpha - test_delta`, where both terms are calculated
on the independent test batch and `test_delta` is the mean of
`1{Y not in C(X)} - alpha(X)`. The orange value is
`loo_coverage = 1 - alpha_hat_loo - delta_hat_loo`, calculated by LOO on the
calibration data. `test_coverage` is retained in the CSV to check the identity
`test_corrected == test_coverage`. The histogram columns use total calibration
size `N_cal = 2n`, not the size of either half. All panels share one set of
histogram bin edges. An explicit common range can optionally be set with
`experiment.histogram_range: [lower, upper]`.

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

## Corrected coverage estimators

For self-validation, the theoretical curve does not use LOO. Each independent
reference trial redraws the calibration sample `C=(D1,D2)` and one test point
`(X0,Y0)`. It records `alpha_C(X0)` and
`Delta_C(X0) = 1{Y0 not in C_C(X0)} - alpha_C(X0)`, and the plotted target is
`1 - mean(reference alpha) - mean(reference Delta)`. The number of Monte Carlo
draws is controlled by `experiment.reference_trials`.

The dedicated LOO experiments use the estimator below.

For each observation in `D1`, the implementation removes that observation, recomputes its adaptive level, and evaluates its pseudo-error against the independent `D2` quantile. Saved metrics include:

- `alpha_hat`: LOO estimate of `E[alpha_delta(X)]`;
- `delta_hat`: LOO estimate of `E[Delta_delta,n]`;
- `old_proxy`: `1 - alpha_hat`, retained only for comparison;
- `corrected_bound`: `1 - alpha_hat - delta_hat`.

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

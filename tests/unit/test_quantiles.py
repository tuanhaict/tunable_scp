import numpy as np

from tscp.quantiles import conformal_quantile


def test_conformal_quantile_uses_finite_sample_rank():
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    assert conformal_quantile(scores, 0.4) == 3.0


def test_conformal_quantile_returns_infinity_for_rank_n_plus_one():
    assert np.isinf(conformal_quantile(np.array([1.0, 2.0]), 0.01))


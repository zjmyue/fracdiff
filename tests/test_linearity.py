import pytest

import numpy as np
from fracdiff import Fracdiff


params_order = list(np.linspace(0.0, 3.0, 50))
params_seed = [42]
params_n_samples = [100, 1000]
params_n_series = [1, 3]
params_a = [0.1, 2.0, -0.1, -2.0]


def make_X(n_samples, n_series):
    return np.random.randn(n_samples, n_series)


# --------------------------------------------------------------------------------


@pytest.mark.parametrize('order', params_order)
@pytest.mark.parametrize('seed', params_seed)
@pytest.mark.parametrize('n_samples', params_n_samples)
@pytest.mark.parametrize('n_series', params_n_series)
def test_add(order, seed, n_samples, n_series):
    """
    Test `D(X1 + X2) = D(X1) + D(X2)`.
    """
    np.random.seed(seed)
    X1 = make_X(n_samples, n_series)
    X2 = make_X(n_samples, n_series)
    D1 = Fracdiff(order).transform(X1)
    D2 = Fracdiff(order).transform(X2)
    DA = Fracdiff(order).transform(X1 + X2)

    assert np.allclose(D1 + D2, DA, equal_nan=True)


@pytest.mark.parametrize('order', params_order)
@pytest.mark.parametrize('seed', params_seed)
@pytest.mark.parametrize('n_samples', params_n_samples)
@pytest.mark.parametrize('n_series', params_n_series)
@pytest.mark.parametrize('a', params_a)
def test_mul(order, seed, n_samples, n_series, a):
    """
    Test `D(a * X) = a * D(X)`.
    """
    np.random.seed(seed)
    X = make_X(n_samples, n_series)
    D1 = Fracdiff(order).transform(X)
    Da = Fracdiff(order).transform(a * X)

    assert np.allclose(a * D1, Da, equal_nan=True)

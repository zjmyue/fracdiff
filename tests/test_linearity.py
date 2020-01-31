import pytest

import numpy as np
from fracdiff import Fracdiff


params_order = list(np.linspace(0.0, 1.0, 18)) + [1.5, 2.5]
params_seed = [42]
params_n_samples = [10, 100, 1000]
params_n_features = [1, 5]
params_a = [0.1, 2.0, -0.1, -2.0]


def make_X(n_samples, n_features):
    return 100 + np.random.randn(n_samples, n_features).cumsum(axis=0)


# --------------------------------------------------------------------------------


@pytest.mark.parametrize('order', params_order)
@pytest.mark.parametrize('seed', params_seed)
@pytest.mark.parametrize('n_samples', params_n_samples)
@pytest.mark.parametrize('n_features', params_n_features)
def test_add(order, seed, n_samples, n_features):
    """
    Test `D(X1 + X2) = D(X1) + D(X2)`.
    """
    np.random.seed(seed)
    X1 = make_X(n_samples, n_features)
    X2 = make_X(n_samples, n_features)
    D1 = Fracdiff(order).transform(X1)
    D2 = Fracdiff(order).transform(X2)
    DA = Fracdiff(order).transform(X1 + X2)

    assert np.allclose(D1 + D2, DA, equal_nan=True)


@pytest.mark.parametrize('order', params_order)
@pytest.mark.parametrize('seed', params_seed)
@pytest.mark.parametrize('n_samples', params_n_samples)
@pytest.mark.parametrize('n_features', params_n_features)
@pytest.mark.parametrize('a', params_a)
def test_mul(order, seed, n_samples, n_features, a):
    """
    Test `D(a * X) = a * D(X)`.
    """
    np.random.seed(seed)
    X = make_X(n_samples, n_features)
    D1 = Fracdiff(order).transform(X)
    Da = Fracdiff(order).transform(a * X)

    assert np.allclose(a * D1, Da, equal_nan=True)

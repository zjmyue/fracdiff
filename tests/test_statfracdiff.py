import pytest

import numpy as np

from fracdiff._stat import StationarityTester
from fracdiff import Fracdiff, StationaryFracdiff

params_seed = [42]
params_n_samples = [500, 1000]
params_n_features = [1, 3]
params_window = [10, 100]
params_precision = [0.01, 0.001]


def make_stationary(seed, n_samples, n_features):
    np.random.seed(seed)
    return np.random.randn(n_samples, n_features)


def make_nonstationary(seed, n_samples, n_features):
    np.random.seed(seed)
    return np.random.randn(n_samples, n_features).cumsum(axis=0)


def is_stat(x):
    return StationarityTester().is_stationary(x)


# --------------------------------------------------------------------------------


@pytest.mark.parametrize('seed', params_seed)
@pytest.mark.parametrize('n_samples', params_n_samples)
@pytest.mark.parametrize('n_features', params_n_features)
@pytest.mark.parametrize('window', params_window)
@pytest.mark.parametrize('precision', params_precision)
def test_order(seed, n_samples, n_features, window, precision):
    """
    Test if `StationaryFracdiff.order_` is the lowest order to make the
    differentiation stationary for array with `n_features > 1`.
    """
    X = make_nonstationary(seed, n_samples, n_features)

    statfracdiff = StationaryFracdiff(window=window, precision=precision)
    statfracdiff.fit(X)
    order = statfracdiff.order_

    Xd_stat = statfracdiff.transform(X)[window:, :]
    Xd_nonstat = np.concatenate([
        Fracdiff(order[i] - precision, window).transform(X[:, [i]])[window:, :]
        for i in range(n_features)
    ], axis=1)

    for i in range(n_features):
        assert is_stat(Xd_stat[:, i])
        assert not is_stat(Xd_nonstat[:, i])


@pytest.mark.parametrize('seed', params_seed)
@pytest.mark.parametrize('n_samples', params_n_samples)
@pytest.mark.parametrize('window', params_window)
def test_lower_is_stat(seed, n_samples, window):
    """
    Test if `StationarityFracdiff.fit` returns `lower`
    if `lower`th differenciation is already stationary.
    """
    X = make_stationary(seed, n_samples, 1)
    statfracdiff = StationaryFracdiff(window=window, lower=0.0)
    order = statfracdiff.fit(X).order_[0]

    assert order == 0.0


@pytest.mark.parametrize('seed', params_seed)
@pytest.mark.parametrize('n_samples', params_n_samples)
@pytest.mark.parametrize('window', params_window)
def test_upper_is_not_stat(seed, n_samples, window):
    """
    Test if `StationarityFracdiff.fit` returns `np.nan`
    if `upper`th differenciation is still non-stationary.
    """
    X = make_nonstationary(seed, n_samples, 1)
    statfracdiff = StationaryFracdiff(window=window, upper=0.0, lower=-1.0)
    order = statfracdiff.fit(X).order_[0]

    assert np.isnan(order)


@pytest.mark.parametrize('seed', params_seed)
@pytest.mark.parametrize('n_samples', params_n_samples)
@pytest.mark.parametrize('n_features', params_n_features)
@pytest.mark.parametrize('window', params_window)
def test_transform(seed, n_samples, n_features, window):
    """
    Test if `StationaryFracdiff.transform` works
    for array with n_features > 1.
    """
    X = make_nonstationary(seed, n_samples, n_features)

    statfracdiff = StationaryFracdiff(window=window).fit(X)
    order = statfracdiff.order_

    Xd = statfracdiff.transform(X)[window:, :]
    Xd_expected = np.concatenate([
        Fracdiff(order[i], window).transform(X[:, [i]])[window:, :]
        for i in range(n_features)
    ], axis=1)

    assert np.allclose(Xd, Xd_expected, equal_nan=True)

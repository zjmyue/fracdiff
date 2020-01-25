import pytest

import numpy as np

from fracdiff._stat import StationarityTester
from fracdiff import Fracdiff, StationaryFracdiff

list_seed = [42]
list_n_samples = [1000, 10000]
list_window = [10, 100]
list_precision = [0.01, 0.001]


def make_stationary(seed, n_samples):
    np.random.seed(seed)
    return np.random.randn(n_samples, 1)


def make_nonstationary(seed, n_samples):
    np.random.seed(seed)
    return np.random.randn(n_samples, 1).cumsum(axis=0)


def isstat(x):
    return StationarityTester().is_stationary(x)

# --------------------------------------------------------------------------------


@pytest.mark.parametrize('seed', list_seed)
@pytest.mark.parametrize('n_samples', list_n_samples)
@pytest.mark.parametrize('window', list_window)
@pytest.mark.parametrize('precision', list_precision)
def test_order(seed, n_samples, window, precision):
    X = make_nonstationary(seed, n_samples)
    statfracdiff = StationaryFracdiff(window=window, precision=precision)
    order = statfracdiff.fit(X).order_[0]

    Xd_stat = Fracdiff(order, window).transform(X)[window:, 0]
    Xd_nonstat = Fracdiff(order - precision, window).transform(X)[window:, 0]

    assert isstat(Xd_stat)
    assert not isstat(Xd_nonstat)


@pytest.mark.parametrize('seed', list_seed)
@pytest.mark.parametrize('n_samples', list_n_samples)
@pytest.mark.parametrize('window', list_window)
def test_lower_is_stat(seed, n_samples, window):
    """
    Test if `StationarityFracdiff.fit` returns `lower`
    if `lower`th differenciation is already stationary.
    """
    X = make_stationary(seed, n_samples)
    statfracdiff = StationaryFracdiff(window=window, lower=0.0)
    order = statfracdiff.fit(X).order_[0]

    assert order == 0.0

@pytest.mark.parametrize('seed', list_seed)
@pytest.mark.parametrize('n_samples', list_n_samples)
@pytest.mark.parametrize('window', list_window)
def test_upper_is_not_stat(seed, n_samples, window):
    """
    Test if `StationarityFracdiff.fit` returns `np.nan`
    if `upper`th differenciation is still non-stationary.
    """
    X = make_nonstationary(seed, n_samples)
    statfracdiff = StationaryFracdiff(window=window, upper=0.1)
    order = statfracdiff.fit(X).order_[0]

    assert np.isnan(order)


@pytest.mark.parametrize('seed', list_seed)
@pytest.mark.parametrize('n_samples', list_n_samples)
@pytest.mark.parametrize('window', list_window)
def test_transform(seed, n_samples, window):
    """
    Test if `StationaryFracdiff.transform` works.
    """
    X = make_nonstationary(seed, n_samples)
    statfracdiff = StationaryFracdiff(window=window)

    order = statfracdiff.fit(X).order_[0]
    fracdiff = Fracdiff(order=order, window=window)

    Xd1 = statfracdiff.transform(X)
    Xd2 = fracdiff.transform(X)

    assert np.allclose(Xd1, Xd2, equal_nan=True)

import pytest

from itertools import product
import numpy as np
from fracdiff._stat import StationarityTester


@pytest.fixture
def list_seed():
    return (42, )

@pytest.fixture
def list_n_samples():
    return (100, 1000, 10000, )


def test_stationary(list_seed, list_n_samples):
    for seed, n_samples in product(list_seed, list_n_samples):
        print(seed, n_samples)
        np.random.seed(seed)
        gauss = np.random.randn(n_samples)

        pvalue = StationarityTester().pvalue(gauss)
        isstat = StationarityTester().is_stationary(gauss)

        assert pvalue < 0.1
        assert isstat


def test_nonstationary(list_seed, list_n_samples):
    for seed, n_samples in product(list_seed, list_n_samples):
        print(seed, n_samples)
        np.random.seed(seed)
        brown = np.random.randn(n_samples).cumsum()

        pvalue = StationarityTester().pvalue(brown)
        isstat = StationarityTester().is_stationary(brown)

        assert pvalue > 0.1
        assert not isstat

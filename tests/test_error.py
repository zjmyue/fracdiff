import pytest

import numpy as np

from fracdiff import Fracdiff
from fracdiff._stat import StationarityTester


params_invalid_d = [-0.1, -1.0]
params_invalid_window = [0, -10]
params_invalid_method = ['foo']
params_invalid_tol_memory = [0.0, -1.0, 2.0]
params_invalid_tol_coef = [0.0, -1.0, 2.0]

X = np.random.randn(100, 2)


# --------------------------------------------------------------------------------


@pytest.mark.parametrize('d', params_invalid_d)
def test_fracdiff_d(d):
    """
    Invalid d
    """
    with pytest.raises(ValueError):
        Fracdiff(d).transform(X)


@pytest.mark.parametrize('window', params_invalid_window)
def test_fracdiff_window(window):
    with pytest.raises(ValueError):
        Fracdiff(window=window).transform(X)


def test_fracdiff_noparams():
    with pytest.raises(ValueError):
        fracdiff = Fracdiff(window=None, tol_coef=None, tol_memory=None)
        fracdiff.transform(X)


@pytest.mark.parametrize('tol_memory', params_invalid_tol_memory)
def test_fracdiff_tol_memory(tol_memory):
    with pytest.raises(ValueError):
        fracdiff = Fracdiff(tol_memory=tol_memory)
        fracdiff.transform(X)


@pytest.mark.parametrize('tol_coef', params_invalid_tol_coef)
def test_fracdiff_tol_coef(tol_coef):
    with pytest.raises(ValueError):
        fracdiff = Fracdiff(tol_coef=tol_coef)
        fracdiff.transform(X)


@pytest.mark.parametrize('method', params_invalid_method)
def test_stat_method(method):
    with pytest.raises(ValueError):
        StationarityTester(method=method).pvalue(X)

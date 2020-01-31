import pytest

from math import floor
import numpy as np
from fracdiff import Fracdiff

params_d = list(np.linspace(0.0, 5.0, 50))
params_tol_memory = [0.5]
params_tol_coef = [0.5]
params_parameter = ['d', 'window', 'tol_memory', 'tol_coef']

LARGE_NUMBER = 2 ** 12
X = np.zeros((LARGE_NUMBER, 2))


def last_coef(d, window):
    return Fracdiff(d, window=window)._fit().coef_[-1]


def lost_memory(d, window):
    coef = Fracdiff(d, window=LARGE_NUMBER)._fit().coef_
    return np.sum(coef[window + 1:])


# --------------------------------------------------------------------------------


@pytest.mark.parametrize('d', params_d)
@pytest.mark.parametrize('tol_memory', params_tol_memory)
def test_tol_memory(d, tol_memory):
    fracdiff = Fracdiff(d, window=None, tol_memory=tol_memory)
    fracdiff.transform(X)
    window = fracdiff.window_

    if d > 1:
        d -= floor(d)
    assert abs(lost_memory(d, window)) < abs(tol_memory)


@pytest.mark.parametrize('d', params_d)
@pytest.mark.parametrize('tol_coef', params_tol_coef)
def test_tol_coef(d, tol_coef):
    fracdiff = Fracdiff(d, window=None, tol_coef=tol_coef)
    fracdiff.transform(X)
    window = fracdiff.window_

    if d.is_integer():
        assert window == d + 1
    else:
        if d > 1:
            d -= floor(d)
        assert abs(last_coef(d, window)) < abs(tol_coef)

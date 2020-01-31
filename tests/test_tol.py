import pytest

import numpy as np
from fracdiff import Fracdiff

params_d = list(np.linspace(0.0, 1.0, 10))
params_tol_memory = [0.1]
params_tol_coef = [0.1]
params_parameter = ['d', 'window', 'tol_memory', 'tol_coef']

LARGE_NUMBER = 2 ** 20
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
        assert abs(last_coef(d, window)) < abs(tol_coef)


# @pytest.mark.parametrize('parameter', params_parameter)
# def test_reset(parameter):
#     pass
#     fracdiff = Fracdiff(0.42, window=42, tol_memory=0.42, tol_coef=0.42)
#     fracdiff.transform(X)
#     window = fracdiff.window_
#     coef = fracdiff.coef_

#     setattr(fracdiff, parameter, 0.424242)
#     fracdiff.transform(X)

#     assert not hasattr(fracdiff, 'window_')

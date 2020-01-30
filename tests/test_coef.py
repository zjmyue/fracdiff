import pytest
from ._coef import get_coefs

import numpy as np
from fracdiff import Fracdiff


params_window = [10]
params_d = list(np.linspace(0.0, 3.0, 31))
params_n_series = [1, 3]
params_n_terms = [10]


def make_X(window, n_terms, n_series):
    """
    Returns
    -------
    np.array([
        [0.0  0.0  ...  0.0  1.0  0.0  ...  0.0] |
        [0.0  0.0  ...  0.0  1.0  0.0  ...  0.0] | n_series
        [0.0  0.0  ...  0.0  1.0  0.0  ...  0.0] |
    ])
         <---- window ---->  <---- n_terms ---->
    """
    return np.concatenate([
        np.zeros((window, n_series)),
        np.ones((1, n_series)),
        np.zeros((n_terms - 1, n_series)),
    ], axis=0)


# --------------------------------------------------------------------------------


@pytest.mark.parametrize('d', params_d)
@pytest.mark.parametrize('window', params_window)
@pytest.mark.parametrize('n_terms', params_n_terms)
def test_coef_attr(d, window, n_terms):
    coef = Fracdiff(d, window=window)._fit().coef
    coef_expected = get_coefs(d, n_terms)

    assert np.allclose(coef, coef_expected)


@pytest.mark.parametrize('d', params_d)
@pytest.mark.parametrize('window', params_window)
@pytest.mark.parametrize('n_terms', params_n_terms)
@pytest.mark.parametrize('n_series', params_n_series)
def test_coef(d, window, n_terms, n_series):
    """
    Test the correctness of coefficients.
    """
    # TODO shift the position of 1
    X = make_X(window, n_terms, n_series)
    Xd = Fracdiff(d, window=window).transform(X)
    coefs_expected = get_coefs(d, n_terms)

    for i in range(n_series):
        assert np.allclose(Xd[window:, i], coefs_expected)

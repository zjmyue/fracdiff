import pytest

import numpy as np
from fracdiff import Fracdiff

params_d = list(np.linspace(0.0, 2.0, 20))
params_window = [10]
params_n_blanks_1 = np.arange(4)
params_n_blanks_2 = np.arange(4)
params_n_terms = [10]
params_n_series = [1, 3]


def make_X(window, n_blanks_1, n_blanks_2, n_terms, n_series):
    """
    Returns
    -------
    np.array([
        [0.0  0.0  ...  0.0] +
        [...  ...  ...  ...] | window
        [0.0  0.0  ...  0.0] +
        [0.0  0.0  ...  0.0] +
        [...  ...  ...  ...] | n_blanks_1
        [0.0  0.0  ...  0.0] +
        [1.0  1.0  ...  1.0]
        [0.0  0.0  ...  0.0] +
        [...  ...  ...  ...] | n_blanks_2
        [0.0  0.0  ...  0.0] +
        [0.0  0.0  ...  0.0] +
        [...  ...  ...  ...] | n_terms - 1
        [0.0  0.0  ...  0.0] +
    ])
         <--- n_series --->
    """
    return np.concatenate([
        np.zeros((window, n_series)),
        np.zeros((n_blanks_1, n_series)),
        np.ones((1, n_series)),
        np.zeros((n_blanks_2, n_series)),
        np.zeros((n_terms, n_series)),
    ], axis=0)


# --------------------------------------------------------------------------------


@pytest.mark.parametrize('d', params_d)
@pytest.mark.parametrize('window', params_window)
@pytest.mark.parametrize('n_blanks_1', params_n_blanks_1)
@pytest.mark.parametrize('n_blanks_2', params_n_blanks_2)
@pytest.mark.parametrize('n_terms', params_n_terms)
@pytest.mark.parametrize('n_series', params_n_series)
def test_coef(d, window, n_blanks_1, n_blanks_2, n_terms, n_series):
    """
    Test the correctness of coefficients.
    """
    X = make_X(window, n_blanks_1, n_blanks_2, n_terms, n_series)

    fracdiff = Fracdiff(d, window=window)
    Xd = fracdiff.transform(X)

    coef_expected = fracdiff.coef

    for i in range(n_series):
        coef = Xd[window + n_blanks_1:, i][:n_terms]
        assert np.allclose(coef, coef_expected)


@pytest.mark.parametrize('d', params_d)
@pytest.mark.parametrize('window', [10])
@pytest.mark.parametrize('n_blanks_1', [0])
@pytest.mark.parametrize('n_blanks_2', [0])
@pytest.mark.parametrize('n_terms', [10])
@pytest.mark.parametrize('n_series', [3])
def test_transform_twice(d, window, n_blanks_1, n_blanks_2, n_terms, n_series):
    """
    Test the correctness of coefficients.
    """
    X = make_X(window, n_blanks_1, n_blanks_2, n_terms, n_series)

    fracdiff = Fracdiff(d, window=window)
    Xd1 = fracdiff.transform(X)
    Xd2 = fracdiff.transform(X)

    assert np.allclose(Xd1, Xd2, equal_nan=True)


@pytest.mark.parametrize('d', params_d)
@pytest.mark.parametrize('window', [10])
@pytest.mark.parametrize('n_blanks_1', [0])
@pytest.mark.parametrize('n_blanks_2', [0])
@pytest.mark.parametrize('n_terms', [10])
@pytest.mark.parametrize('n_series', [3])
def test_change_d(d, window, n_blanks_1, n_blanks_2, n_terms, n_series):
    """
    Test the correctness of coefficients.
    """
    X = make_X(window, n_blanks_1, n_blanks_2, n_terms, n_series)

    fracdiff = Fracdiff(0.42, window=window)
    _ = fracdiff.transform(X)

    fracdiff.d = d

    Xd = fracdiff.transform(X)
    Xd_expected = Fracdiff(d, window=window).transform(X)

    assert np.allclose(Xd, Xd_expected, equal_nan=True)


@pytest.mark.parametrize('d', params_d)
@pytest.mark.parametrize('window', [10])
@pytest.mark.parametrize('n_blanks_1', [0])
@pytest.mark.parametrize('n_blanks_2', [0])
@pytest.mark.parametrize('n_terms', [10])
@pytest.mark.parametrize('n_series', [3])
def test_change_window(d, window, n_blanks_1, n_blanks_2, n_terms, n_series):
    """
    Test the correctness of coefficients.
    """
    X = make_X(window, n_blanks_1, n_blanks_2, n_terms, n_series)

    fracdiff = Fracdiff(d, window=1)
    _ = fracdiff.transform(X)

    fracdiff.window = window

    Xd = fracdiff.transform(X)
    Xd_expected = Fracdiff(d, window=window).transform(X)

    assert np.allclose(Xd, Xd_expected, equal_nan=True)

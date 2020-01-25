import pytest

import numpy as np
from fracdiff import Fracdiff


windows = [4, 100, 1000]

# order of differentiation and corresponding coefficients
_dict_coeff = {
    0.0: [1.0, 0.0, 0.0, 0.0],
    1.0: [1.0, -1.0, 0.0, 0.0],
    1.0 / 2.0: [1.0, -1.0 / 2.0, -1.0 / 8.0, -1.0 / 16.0],
    1.0 / 3.0: [1.0, -1.0 / 3.0, -1.0 / 9.0, -5.0 / 81.0],
    1.0 / 4.0: [1.0, -1.0 / 4.0, -3.0 / 32.0, -7.0 / 128.0],
    1.0 / 5.0: [1.0, -1.0 / 5.0, -2.0 / 25.0, -6.0 / 125.0],
    1.0 / 6.0: [1.0, -1.0 / 6.0, -5.0 / 72.0, -55.0 / 1296.0],
    1.0 / 7.0: [1.0, -1.0 / 7.0, -3.0 / 49.0, -13.0 / 343.0],
    1.0 / 8.0: [1.0, -1.0 / 8.0, -7.0 / 128.0, -35.0 / 1024.0],
    1.0 / 9.0: [1.0, -1.0 / 9.0, -4.0 / 81.0, -68.0 / 2187.0],
}

orders_coeffs = [
    (order, np.array(coeff).reshape(-1, 1))
    for order, coeff in _dict_coeff.items()
]


# --------------------------------------------------------------------------------


def make_X(window):
    """
    Returns
    -------
    np.array([[0, 0, ..., 0, 0, 1, 0, 0, 0, 0]])
               <--- window --->
    """
    zeros = np.zeros(window)
    delta = np.array([1.0, 0.0, 0.0, 0.0])
    return np.concatenate([zeros, delta], axis=0).reshape(-1, 1)


@pytest.mark.parametrize('window', windows)
@pytest.mark.parametrize('order, coeff', orders_coeffs)
def test_coeff(window, order, coeff):
    """
    Test the correctness of coefficients.
    """
    X = make_X(window)
    Xd = Fracdiff(order, window=window).transform(X)

    assert np.allclose(Xd[window:, :], coeff)


@pytest.mark.parametrize('window', windows)
@pytest.mark.parametrize('order, coeff', orders_coeffs)
def test_coeff_multiple(window, order, coeff):
    """
    Test fracdiff of array with n_features > 1.
    """
    window = 100
    X = np.concatenate([make_X(window), make_X(window), make_X(window)], axis=1)
    Xd = Fracdiff(order, window=window).transform(X)

    coeffs = np.concatenate([coeff, coeff, coeff], axis=1)

    assert np.allclose(Xd[window:, :], coeffs)

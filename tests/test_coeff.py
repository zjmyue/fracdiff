import pytest

import numpy as np
from fracdiff import Fracdiff


list_window = [4, 10, 100, 1000]
list_n_features = [1, 5]

# order of differentiation and corresponding coefficients
_dict_coef = {
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

orders_coefs = [
    (order, np.array(coef))
    for order, coef in _dict_coef.items()
]


def make_X(window, n_features):
    """
    Returns
    -------
    np.array([[0.0  0.0  ...  0.0  1.0  0.0  0.0  0.0]])
               <------- window ------>
    """
    x = np.concatenate([
        np.zeros(window - 1),
        np.array([1]),
        np.zeros(3),
    ], axis=0)

    return np.stack([x for _ in range(n_features)], axis=1)


# --------------------------------------------------------------------------------


@pytest.mark.parametrize('window', list_window)
@pytest.mark.parametrize('n_features', list_n_features)
@pytest.mark.parametrize('order, coef', orders_coefs)
def test_coef(window, order, coef, n_features):
    """
    Test the correctness of coefficients.
    """
    X = make_X(window, n_features)
    Xd = Fracdiff(order, window=window).transform(X)

    for i in range(n_features):
        assert np.allclose(Xd[-4:, i], coef)


# @pytest.mark.parametrize('order, _', orders_coefs)
# @pytest.mark.parametrize('window', windows)
# def test_cache(order, window, _):
#     fracdiff = Fracdiff(order, window=window)

#     coefs = fracdiff.coef
#     coefs_cached = fracdiff.coef

#     assert np.equal(coefs, coefs_cached).all()

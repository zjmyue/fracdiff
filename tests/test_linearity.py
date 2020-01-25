import pytest

from itertools import product
import numpy as np
from fracdiff import Fracdiff


orders = list(np.linspace(0.0, 2.0, 11))

list_X1 = [
    1.0 + np.random.randn(100, 1),
    1.0 + np.random.randn(100, 1).cumsum(axis=1),
]

list_X2 = [
    1.0 + np.random.randn(100, 1),
    1.0 + np.random.randn(100, 1).cumsum(axis=1),
]

list_a = [0.1, 0.5, 2.0, 10.0, -0.1, -0.5, -2.0, -10.0]


# --------------------------------------------------------------------------------


@pytest.mark.parametrize('order', orders)
@pytest.mark.parametrize('X1', list_X1)
@pytest.mark.parametrize('X2', list_X2)
def test_add(order, X1, X2):
    """
    Test `D(X1 + X2) = D(X1) + D(X2)`.
    """
    D1 = Fracdiff(order).transform(X1)
    D2 = Fracdiff(order).transform(X2)
    DA = Fracdiff(order).transform(X1 + X2)

    assert np.allclose(D1 + D2, DA, equal_nan=True)


@pytest.mark.parametrize('order', orders)
@pytest.mark.parametrize('X1', list_X1)
@pytest.mark.parametrize('a', list_a)
def test_mul(order, X1, a):
    """
    Test `D(a * X1) = a * D(X1)`.
    """
    D1 = Fracdiff(order).transform(X1)
    DM = Fracdiff(order).transform(a * X1)

    assert np.allclose(a * D1, DM, equal_nan=True)


@pytest.mark.parametrize('order', orders)
@pytest.mark.parametrize('X1', list_X1)
@pytest.mark.parametrize('X2', list_X2)
def test_add_multiple(order, X1, X2):
    """
    Test `D(X1 + X2) = D(X1) + D(X2)` for X1, X2 with n_features > 1.
    """
    X1_ = np.concatenate([X1, X2, X1], axis=1)
    X2_ = np.concatenate([X2, X1, X2], axis=1)

    D1 = Fracdiff(order).transform(X1_)
    D2 = Fracdiff(order).transform(X2_)
    DA = Fracdiff(order).transform(X1_ + X2_)

    assert np.allclose(D1 + D2, DA, equal_nan=True)


@pytest.mark.parametrize('order', orders)
@pytest.mark.parametrize('X1', list_X1)
@pytest.mark.parametrize('a', list_a)
def test_mul_multiple(order, X1, a):
    """
    Test `D(a * X1) = a * D(X1). for X1 with n_features > 1`.
    """
    X1_ = np.concatenate([X1, X1, X1])
    D1 = Fracdiff(order).transform(X1_)
    DM = Fracdiff(order).transform(a * X1_)

    assert np.allclose(a * D1, DM, equal_nan=True)

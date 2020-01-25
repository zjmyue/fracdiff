import pytest

from itertools import product
import numpy as np
from fracdiff import Fracdiff


@pytest.fixture
def orders():
    return np.linspace(0.0, 2.0, 21)

@pytest.fixture
def list_X1():
    np.random.seed(1)
    return (
        1.0 + np.random.randn(100, 1),
        1.0 + np.random.randn(100, 1),
        1.0 + np.random.randn(100, 1).cumsum(axis=1),
        1.0 + np.random.randn(100, 1).cumsum(axis=1),
    )

@pytest.fixture
def list_X2():
    np.random.seed(2)
    return (
        1.0 + np.random.randn(100, 1),
        1.0 + np.random.randn(100, 1).cumsum(axis=1),
        1.0 + np.random.randn(100, 1),
        1.0 + np.random.randn(100, 1).cumsum(axis=1),
    )

@pytest.fixture
def list_a():
    return (0.1, 0.5, 2.0, 10.0, -0.1, -0.5, -2.0, -10.0, )

# --------------------------------------------------------------------------------

def test_add(orders, list_X1, list_X2):
    """
    Test `D(X1 + X2) = D(X1) + D(X2)`.
    """
    for order, (X1, X2) in product(orders, zip(list_X1, list_X2)):
        D1 = Fracdiff(order).transform(X1)
        D2 = Fracdiff(order).transform(X2)
        DA = Fracdiff(order).transform(X1 + X2)

        assert np.allclose(D1 + D2, DA, equal_nan=True)


def test_mul(orders, list_X1, list_a):
    """
    Test `D(a * X1) = a * D(X1).`
    """
    for order, X1, a in product(orders, list_X1, list_a):
        D1 = Fracdiff(order).transform(X1)
        DM = Fracdiff(order).transform(a * X1)

        assert np.allclose(a * D1, DM, equal_nan=True)

import pytest

import numpy as np

from fracdiff import Fracdiff
from fracdiff._stat import StationarityTester


invalid_orders = ['hoge']
invalid_windows = ['hoge', -10]
invalid_methods = ['hoge']


# --------------------------------------------------------------------------------


# @pytest.mark.parametrize('order', invalid_orders)
# def test_fracdiff_order(order):
#     X = np.random.randn(100, 1)
#     with pytest.raises(ValueError):
#         Fracdiff(order=order).transform(X)


# @pytest.mark.parametrize('window', invalid_windows)
# def test_fracdiff_window(window):
#     X = np.random.randn(100, 1)
#     with pytest.raises(ValueError):
#         Fracdiff(window=window).transform(X)


def test_fracdiff_noparams():
    X = np.random.randn(100)
    with pytest.raises(ValueError):
        fracdiff = Fracdiff(window=None, tol_coef=None, tol_memory=None)
        fracdiff.transform(X)


@pytest.mark.parametrize('method', invalid_methods)
def test_stat_method(method):
    X = np.random.randn(100)
    with pytest.raises(ValueError):
        StationarityTester(method=method).pvalue(X)

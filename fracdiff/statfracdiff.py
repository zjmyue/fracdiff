from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_array
import numpy as np

from .fracdiff import Fracdiff
from ._stat import StationarityTester


class StationaryFracdiff(TransformerMixin):
    """
    Carry out fractional derivative with the minumum order
    with which the differentiation becomes stationary.

    Parameters
    ----------
    - stationarity_test : {'ADF'}, default 'ADF'
        Method of stationarity test.
    - pvalue : float, default .05
        P-value to judge stationarity.
    - precision : float, default .01
        Precision for the order of differentiation.
    - upper : float, default 1.0
        Upper limit of the range to search the order.
    - lower : float, default 0.0
        Lower limit of the range to search the order.
    - window : positive int or -1, default 10
        Window to compute differentiation.
        If -1, ...

    Attributes
    ----------
    - order_ : array-like, shape (n_features, )
        Minimum order of fractional differentiation
        that makes time-series stationary.

    Note
    ----
    If `upper`th differentiation of series is still non-stationary,
    order_ is set to `np.nan`.
    If `lower`th differentiation of series is already stationary,
    order_ is set to `lower`, but the true value may be smaller.
    """
    def __init__(
        self,
        stat_method='ADF',
        pvalue=.05,
        precision=.01,
        upper=1.0,
        lower=0.0,
        window=10
    ):
        self.stat_method = stat_method
        self.pvalue = pvalue
        self.precision = precision
        self.upper = upper
        self.lower = lower
        self.window = window

    def fit(self, X, y=None):
        self.order_ = self.__search_order(X)
        return self

    def transform(self, X, y=None):
        check_array(X)
        _, n_features = X.shape

        return np.concatenate([
            Fracdiff(self.order_[i], window=self.window).transform(X[:, [i]])
            for i in range(n_features)
        ], axis=1)

    def __search_order(self, X):
        """
        Carry out binary search of minimum order of fractional
        differentiation to make the time-series stationary.

        Parameters
        ----------
        X : array-like, shape (n_samples, )
        """
        _, n_features = X.shape
        if n_features > 1:
            return np.concatenate([
                self.__search_order(X[:, :1]),
                self.__search_order(X[:, 1:]),
            ], axis=0)

        tester = StationarityTester(method=self.stat_method)

        Xu = Fracdiff(self.upper, window=self.window).transform(X)
        if not tester.is_stationary(Xu[self.window:, 0], pvalue=self.pvalue):
            return np.array([np.nan])
        Xl = Fracdiff(self.lower, window=self.window).transform(X)
        if tester.is_stationary(Xl[self.window:, 0], pvalue=self.pvalue):
            return np.array([self.lower])

        upper, lower = self.upper, self.lower
        while upper - lower > self.precision:
            m = (upper + lower) / 2
            Xm = Fracdiff(m, window=self.window).transform(X)
            if tester.is_stationary(Xm[self.window:, 0], pvalue=self.pvalue):
                upper = m
            else:
                lower = m

        return np.array([upper])

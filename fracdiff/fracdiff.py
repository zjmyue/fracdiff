from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
import numpy as np

from sklearn.utils.estimator_checks import check_estimator

from ._stat import StationarityTester:

class Fracdiff(BaseEstimator, TransformerMixin):
    """
    Carry out fractional differentiation.

    Parameters
    ----------
    - order : float, default 1.0
        Order of differentiation.  Must be 0-1
    - window : positive int or -1, default 10
        Window to compute differentiation.
        If -1, ...

    Examples
    --------
    >>> fracdiff = Fracdiff(order=0.5)
    >>> X = np.array([...])
    >>> X_d = fracdiff.transform(X)
    >>> X_d
    """
    def __init__(self, order=1.0, window=10):
        """Initialize self."""
        self.order = order
        self.window = window

    @staticmethod
    def __coeff(order, window):
        def omega(order, window):
            c = 1.0
            for k in range(window + 1):
                yield c
                c *= (k - order) / (k + 1)
        return np.array(list(omega(order, window)))[::-1]

    def fit(self, X, y=None):
        """
        Parameters
        ----------
        - X : array-like, shape (n_samples, n_features)
            Time-series to differentiate.
        - y : None
            Ignored.

        Returns
        -------
        self
        """
        X = check_array(X)
        self.n_features_ = X.shape[1]
        return self

    def __check_order(self):
        """Check if the value of order is sane"""
        if not (isinstance(self.order, float) or isinstance(self.order, int)):
            raise TypeError('order must be int or float.')

    def __check_window(self):
        """Check if the value of window is sane"""
        if not isinstance(self.window, int):
            raise TypeError('window must be int.')
        if not (self.window == -1 or self.window > 0):
            raise ValueError('window must be -1 or positive integer.')

    def transform(self, X):
        """
        Perform fractional differentiation on array.

        Parameters
        ----------
        - X : array-like, shape (n_samples, )
            Time-series to differentiate.

        Returns
        -------
        - X_differentiated : array-like, shape (n_samples, )
            Differentiated time-series.
        """
        # self.__check_order()
        # self.__check_window()
        check_is_fitted(self, 'n_features_')
        X = check_array(X, estimator=self)
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')

        __max_window = 100  # TODO TBD
        n_samples = X.shape[0]
        window = self.window if self.window != -1 else __max_window

        coeff = self.__class__.__coeff(self.order, window)
        d = np.r_[coeff, np.zeros(n_samples - 1)]
        D_extended = np.vstack([
            np.roll(d, shift) for shift in range(n_samples)
        ])
        D = D_extended[:, window:]

        X_d = D @ X

        if self.window != -1:
            X_d[:window] = np.nan

        return X_d


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
    - window : positive int or -1, default 10
        Window to compute differentiation.
        If -1, ...

    Attributes
    ----------
    - order_ : float
        Minimum order of fractional differentiation
        that makes time-series stationary.
    - fracdiff_ : Fracdiff
        Fracdiff object.
    """
    def __init__(self,
                 stationarity_test='ADF',
                 pvalue=.05,
                 precision=.01,
                 window=3):
        self.tester = StationarityTester(method=stationarity_test)
        self.pvalue = pvalue
        self.precision = precision
        self.window = window

    def __binary_search_order(self, X, lower, upper):
        """
        Carry out binary search of minimum order of fractional
        differentiation to make the time-series stationary.
        """
        tester = self.tester

        X_u = Fracdiff(upper, window=self.window).transform(X)[self.window:]
        if not tester.is_stationary(X_u, pvalue=self.pvalue):
            return np.nan
        X_l = Fracdiff(lower, window=self.window).transform(X)[self.window:]
        if tester.is_stationary(X_l, pvalue=self.pvalue):
            return lower

        while upper - lower > self.precision:
            m = (upper + lower) / 2
            X_m = Fracdiff(m, window=self.window).transform(X)[self.window:]
            if tester.is_stationary(X_m, pvalue=self.pvalue):
                upper = m
            else:
                lower = m
        return upper

    def fit(self, X, y=None):
        self.order_ = self.__binary_search_order(X, lower=0.0, upper=1.0)
        self.fracdiff_ = Fracdiff(self.order_, self.window)
        return self

    def transform(self, X, y=None):
        return self.fracdiff_.transform(X, y)


check_estimator(Fracdiff)
check_estimator(StationaryFracdiff)

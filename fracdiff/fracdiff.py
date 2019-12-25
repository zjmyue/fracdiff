from sklearn.base import TransformerMixin
from statsmodels.tsa.stattools import adfuller
import numpy as np


class FracDiff(TransformerMixin):
    """
    Carry out fractional differentiation.

    Parameters
    ----------
    - order : float
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
    @staticmethod
    def __check_order(order):
        """Check if the given value of order is sane"""
        if not (isinstance(order, float) or isinstance(order, int)):
            raise TypeError('order must be int or float.')

    @staticmethod
    def __check_window(window):
        """Check if the given value of window is sane"""
        if not isinstance(window, int):
            raise TypeError('window must be int.')
        if not (window == -1 or window > 0):
            raise ValueError('window must be -1 or positive integer.')

    def __init__(self, order, window=100):
        """Initialize self."""
        # TODO default window = 100 sensible?
        self.__class__.__check_order(order)
        self.__class__.__check_window(window)

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
        return self

    def transform(self, X, y=None):
        """
        Perform fractional differentiation on array.

        Parameters
        ----------
        - X : array-like, shape (n_samples, )
            Time-series to differentiate.
        - y : None
            Ignored.

        Returns
        -------
        - X_d : array-like, shape (n_samples, )
            Differentiated time-series.
        """
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


class StationarityTester:
    """
    Carry out stationarity test of time-series.

    Parameters
    ----------
    - method : {'ADF'}, default 'ADF'
        'ADF' : Augmented Dickey-Fuller unit-root test.
    """
    @staticmethod
    def __check_method(method):
        if method not in ('ADF', ):
            raise ValueError(f'Invalid method: {method}')

    def __init__(self, method='ADF'):
        self.__class__.__check_method(method)
        self.method = method

    def score(self, X, y=None, value='pvalue'):
        """
        Return p-value of stationarity test.

        Parameters
        ----------
        - X : array-like, shape (n_samples, )
            Time-series to score p-value.
        - y : None
            Ignored.
        - value : {'pvalue', 'statistics', 'all'}, default 'pvalue'
            'pvalue' : p-value.
            'statistics' : statistics of unit-root test.
            'all' : All return values from statsmodels.

        Returns
        -------
        pvalue : float
            p-value of stationarity test.
        """
        if self.method == 'ADF':
            if value == 'pvalue':
                _, pvalue, _, _, _, _ = adfuller(X)
                return pvalue
            if value == 'statistics':
                statistics, _, _, _, _, _ = adfuller(X)
                return statistics
            if value == 'all':
                return adfuller(X)
            raise ValueError()

    def is_stationary(self, X, y=None, pvalue=.05):
        """
        Return if stationarity test implies stationarity.

        Returns
        -------
        is_stationary : bool
            True means that the null hypothes that a unit-root is present
            has been rejected.
        """
        if self.method == 'ADF':
            return self.score(X) < pvalue


class StationaryFracDiff(TransformerMixin):
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
    - fracdiff_ : FracDiff
        FracDiff object.
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

        X_u = FracDiff(upper, window=self.window).transform(X)[self.window:]
        if not tester.is_stationary(X_u, pvalue=self.pvalue):
            return np.nan
        X_l = FracDiff(lower, window=self.window).transform(X)[self.window:]
        if tester.is_stationary(X_l, pvalue=self.pvalue):
            return lower

        while upper - lower > self.precision:
            m = (upper + lower) / 2
            X_m = FracDiff(m, window=self.window).transform(X)[self.window:]
            if tester.is_stationary(X_m, pvalue=self.pvalue):
                upper = m
            else:
                lower = m
        return upper

    def fit(self, X, y=None):
        self.order_ = self.__binary_search_order(X, lower=0.0, upper=1.0)
        self.fracdiff_ = FracDiff(self.order_, self.window)
        return self

    def transform(self, X, y=None):
        return self.fracdiff_.transform(X, y)

from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_array
import numpy as np

from ._stat import StationarityTester


class Fracdiff:
    """
    Carry out fractional differentiation.

    Parameters
    ----------
    - order : float, default 1.0
        Order of differentiation.
    - window : positive int or -1, default 10
        Window to compute differentiation.
        If -1, ...

    Examples
    --------
    >>> fracdiff = Fracdiff(order=0.5)
    >>> X = np.array([1., 0., 0., 0., 0., 0.]).reshape(-1, 1)
    >>> Xd = fracdiff.transform(X)
    >>> Xd
    array([[ 1.        ],
           [-0.5       ],
           [-0.125     ],
           [-0.0625    ],
           [-0.0390625 ],
           [-0.02734375]])
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
        - X_d : array-like, shape (n_samples, )
            Differentiated time-series.
        """
        self.__check_order()
        self.__check_window()
        X = check_array(X, estimator=self)

        n_samples, n_features = X.shape
        if n_features > 1:
            return np.hstack([
                self.transform(X[:, :1]),
                self.transform(X[:, 1:]),
            ])

        __max_window = 100  # TODO TBD
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
    - order_ : array-like, shape (n_features, )
        Minimum order of fractional differentiation
        that makes time-series stationary.
    - fracdiff_ : Fracdiff
        Fracdiff object.
    """
    def __init__(self,
                 stat_method='ADF',
                 pvalue=.05,
                 precision=.01,
                 upper=1.0,
                 lower=0.0,
                 window=10):
        self.stat_method = stat_method
        self.pvalue = pvalue
        self.precision = precision
        self.upper = upper
        self.lower = lower
        self.window = window

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
            return np.hstack([
                self.__search_order(X[:, :1]),
                self.__search_order(X[:, 1:]),
            ])

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

    def fit(self, X, y=None):
        self.order_ = self.__search_order(X)
        return self

    def transform(self, X, y=None):
        _, n_features = X.shape

        return np.hstack([
            (
                Fracdiff(self.order_[i], window=self.window)
                .transform(X[:, i].reshape(-1, 1))
            )
            for i in range(n_features)
        ])

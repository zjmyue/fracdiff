from copy import copy
from functools import partial
from bisect import bisect

import numpy as np
from sklearn.utils.validation import check_array
from scipy.special import binom


class Fracdiff:
    """
    Carry out fractional differentiation.

    Parameters
    ----------
    - d : float, default 1.0
        Order of differentiation.
    - window : int > 0 or None, default 10
        Minimum number of observations to evaluate each term in fracdiff.
        If None, it will be determined according to `tol_memory` or `tol_coef`.
    - tol_memory : float in (0, 1) or None, default None
        Tolerance of memory loss to determine `window`.
        That is, `window` is chosen as the minimum integer that makes the
        absolute value of the sum of fracdiff coefficients from `window + 1`th
        term is smaller than `tol_memory`.
        If `window` is specified, ignored.
    - tol_coef : float in (0, 1) or None, default None
        Tolerance of memory loss to determine `window`.
        That is, `window` is chosen as the minimum integer that makes the
        absolute value of `window + 1`th fracdiff coefficient is smaller
        than `tol_memory`.
        If `window` is specified, ignored.
    - window_policy : {'fixed'}, default 'fixed'
        If fixed :
            Each term in fracdiff time-series will be evaluated using
            `window` observations.
            In other words, in a fracdiff operator as a polynominal of a
            backshift operator, the sequence of coefficient will always be
            truncated up to the `window`th term.
            The beginning `window - 1` terms in fracdiff time-series will be
            filled with `numpy.nan`.
        If expanding (not available yet) :
            Each term in fracdiff time-series will be evaluated using at least
            `window` observations.
            The beginning `window - 1` terms in fracdiff time-series will be
            filled with `numpy.nan`.
    - max_window : int, default 2 ** 12
        Maximum value of window when determining it from `tol_memory`
        and/or `tol_coef`.

    Attributes
    ----------
    - window_ : int
        Minimum number of observations to evaluate each term in fracdiff.
        If `window` is specified, it is set to it.
        If `window` is None, it will be determined based on `tol_memory`
        and/or `tol_coef`.
    - coef_ : array, shape (window, )
        Sequence of coefficients in fracdiff operator.

    Notes
    -----
    window will become extremely large for small tol bla bla bla.

    Examples
    --------
    >>> fracdiff = Fracdiff(0.5)
    >>> X = ...
    >>> fracdiff.transform(X)
    ...
    >>> fracdiff.coef
    ...
    """
    def __init__(
        self,
        d=1.0,
        window=10,
        tol_memory=None,
        tol_coef=None,
        window_policy='fixed',
        max_window=2 ** 12,
    ):
        self.d = d
        self.window = window
        self.tol_memory = tol_memory
        self.tol_coef = tol_coef
        self.window_policy = window_policy
        self.max_window = max_window

    def transform(self, X):
        """
        Return fractional differentiation of array.

        Parameters
        ----------
        - X : array-like, shape (n_samples, n_series)
            Time-series to perform fractional differentiation.

        Returns
        -------
        - X_d : array, shape (n_samples, n_series)
            Fractionally differentiated time-series.
            The beginning `self.window - 1` terms will be filled with
            `numpy.nan`.
        """
        self._fit()

        X = check_array(X, estimator=self)
        n_samples, n_series = X.shape

        D = partial(np.convolve, self.coef_, mode='valid')
        Xd = np.apply_along_axis(D, 0, X)

        nans = np.full((n_samples - Xd.shape[0], n_series), np.nan)
        Xd = np.concatenate([nans, Xd], axis=0)

        return Xd

    def _fit(self):
        """
        Set `self.window_` and `self.coef_`.

        Returns
        -------
        self
        """
        self._check_params()

        # If parameters have been changed, reset attributes
        if (
            self.d != getattr(self, '_d', self.d) or
            self.window != getattr(self, '_window', self.window) or
            self.tol_memory != getattr(self, '_tol_memory', self.tol_memory) or
            self.tol_coef != getattr(self, '_tol_coef', self.tol_coef)
        ):
            delattr(self, 'window_')
            delattr(self, 'coef_')

        if not hasattr(self, 'coef_'):
            self.coef_ = self._get_coef()
            self.window_ = self._get_window()
            self.coef_ = self.coef_[:self.window_]

            # Cache parameters with which attributes were computed
            self._d = self.d
            self._window = self.window
            self._tol_memory = self.tol_memory
            self._tol_coef = self.tol_coef

        return self

    def _check_params(self):
        if self.d < 0.0:
            raise ValueError('d must be positive.')

        if self.window is not None:
            if self.window < 1:
                raise ValueError('window must be positive.')

        if self.tol_memory is not None:
            if not 0.0 < self.tol_memory < 1.0:
                raise ValueError('tol_memory must be in (0.0, 1.0).')

        if self.tol_coef is not None:
            if not 0.0 < self.tol_coef < 1.0:
                raise ValueError('tol_coef must be in (0.0, 1.0).')

        if not any([self.window, self.tol_memory, self.tol_coef]):
            raise ValueError(
                'None of window, tol_coef and tol_memory are specified.'
            )

    def _get_coef(self):
        """
        Return array of coefficients.

        Returns
        -------
        coef : array, shape (n_terms, )
            Array of coefficients of fracdiff.
        """
        if self.d >= 1.0:
            return np.diff(self._descendant._get_coef(), prepend=0.0)

        n_terms = self.window or self.max_window
        s = np.tile([1.0, -1.0], -(-n_terms // 2))[:n_terms]

        return s * binom(self.d, np.arange(n_terms))

    def _get_window(self):
        """
        Return window determined by `self.window`, `self.tol_memory`,
        and `self.tol_coef`.

        Returns
        -------
        window : int
        """
        if self.window:
            return self.window

        if self.d.is_integer():
            return int(self.d) + 1

        if self.d > 1:
            return self._descendant._get_window()

        window = max(
            bisect(-np.cumsum(self.coef_), -(self.tol_memory or np.inf)) + 1,
            bisect(-np.abs(self.coef_), -(self.tol_coef or np.inf)) + 1,
        )
        if self.window == self.max_window:
            raise RuntimeWarning(
                f'window saturated with max_window = {self.max_window}.'
            )
        return window

    @property
    def _descendant(self):
        """
        Return `Fracdiff` with `d` one smaller than self.
        Used to evaluate `coef_` and `window_` of Fracdiff with d > 1.

        Returns
        -------
        descendant : Fracdiff
        """
        descendant = copy(self)
        descendant.d = self.d - 1.0
        return descendant

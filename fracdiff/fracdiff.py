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
    - tol_memory : float in [0, 1] or None, default None
        Tolerance of memory loss to determine `window`.
        That is, `window` is chosen as the minimum integer that makes the
        absolute value of the sum of fracdiff coefficients from `window + 1`th
        term is smaller than `tol_memory`.
        If `window` is specified, ignored.
    - tol_coef : float in [0, 1] or None, default None
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

    Attributes
    ----------
    - coef : array, shape (window, )
        Sequence of coefficients in fracdiff operator.

    Examples
    --------
    >>> fracdiff = Fracdiff(0.5)
    >>> X = ...
    >>> fracdiff.transform(X)
    ...
    >>> fracdiff.coef
    ...
    """
    MAX_WINDOW = 2 ** 20  # upper limit of `self.window`.

    def __init__(
        self,
        d=1.0,
        window=10,
        tol_memory=None,
        tol_coef=None,
        window_policy='fixed',
    ):
        """Initialize self."""
        self.d = d
        self.window = window
        self.tol_memory = tol_memory
        self.tol_coef = tol_coef
        self.window_policy = window_policy

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
        X = check_array(X, estimator=self)
        self._fit()
        n_samples, n_series = X.shape

        D = partial(np.convolve, self.coef, mode='valid')
        Xd = np.apply_along_axis(D, 0, X)

        nans = np.full((n_samples - Xd.shape[0], n_series), np.nan)
        Xd = np.concatenate([nans, Xd], axis=0)

        return Xd

    def _fit(self):
        """
        Set `self.window` and `self.coef`.
        """
        # If parameters have been changed, recompute coef
        if (
            getattr(self, '_d', self.d) != self.d
            or getattr(self, '_window', self.window) != self.window
            or getattr(self, '_tol_memory', self.tol_memory) != self.tol_memory
            or getattr(self, '_tol_coef', self.tol_coef) != self.tol_coef
        ):
            delattr(self, 'coef')

        if not hasattr(self, 'coef'):
            self.coef = self._get_coef()
            self._d = self.d
            self._tol_memory = self.tol_memory
            self._tol_coef = self.tol_coef
            self._window = self.window

        return self

    def _get_coef(self):
        """
        Return array of coefficients.

        Note
        ----
        The k-th coefficient (k = 0, 1, 2, ...) is given by ::
            ((-1) ** k) * poch_down(d, k) / k!
        """
        # Check parameters
        if self.d < 0.0:
            raise ValueError('d must be positive.')
        if self.tol_memory is not None:
            if not 0.0 < self.tol_memory < 1.0:
                raise ValueError('tol_memory must be in (0.0, 1.0).')
        if self.tol_coef is not None:
            if not 0.0 < self.tol_coef < 1.0:
                raise ValueError('tol_coef must be in (0.0, 1.0).')

        # Compute
        n_terms = self.window or self.MAX_WINDOW
        k = np.arange(n_terms, dtype=np.float64)
        s = np.tile([1.0, -1.0], -(-n_terms // 2))[:n_terms]
        coef = s * binom(self.d, k)

        # Truncate
        if self.window is None:
            if self.tol_memory is None and self.tol_coef is None:
                raise ValueError(
                    'None of window, tol_coef and tol_memory are specified.'
                )
            # FIXME works only for 0 < d < 1
            lost_memory = 1.0 - np.cumsum(self.coef)
            window = max(
                bisect(-lost_memory, -self.tol_memory or np.inf),
                bisect(-self.coef, -self.tol_coef or np.inf),
            )
            self.window = window
            coef = coef[:self.window]

        if self.window < 1:
            raise ValueError('window should be positive.')

        return coef

from sklearn.utils.validation import check_array
import numpy as np


class Fracdiff:
    """
    Carry out fractional differentiation.

    Parameters
    ----------
    - order : float, default 1.0
        Order of differentiation.
    - window : int > 0 or None, default 10
        Minimum number of observations to evaluate each term in fracdiff.
        If None, it will be determined according to `tol_memory` or `tol_coef`.
    - tol_memory : float in [0, 1] or None, default None
        Tolerance of memory loss to determine `window`.
        That is, `window` is chosen as the minimum integer that makes the
        absolute value of the sum of fracdiff coefficients from `window + 1`th
        term is smaller than `tol_memory`.
        If `width_window` is specified, ignored.
    - tol_coef : float in [0, 1] or None, default None
        Tolerance of memory loss to determine `window`.
        That is, `window` is chosen as the minimum integer that makes the
        absolute value of `window + 1`th fracdiff coefficient is smaller
        than `tol_memory`.
        If `width_window` is specified, ignored.
    - window_policy : {'fixed', 'expanding'}, default 'fixed'
        If fixed :
            Each term in fracdiff time-series will be evaluated using
            `window` observations.
            In other words, in a fracdiff operator as a polynominal of a
            backshift operator, the sequence of coefficient will always be
            truncated up to the `window`th term.
            The beginning `window - 1` terms in fracdiff time-series will be
            filled with `numpy.nan`.
        If expanding :
            Each term in fracdiff time-series will be evaluated using at least
            `window` observations.
            The beginning `window - 1` terms in fracdiff time-series will be
            filled with `numpy.nan`.

    Examples
    --------
    >>> fracdiff = Fracdiff(0.5)
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

    MAX_WINDOW = 10 ** 6  # upper limit of `self.window`.

    def __init__(
        self,
        order=1.0,
        window=10,
        tol_memory=None,
        tol_coef=None,
        window_policy='fixed',
    ):
        """Initialize self."""
        self.order = order
        self.window = window
        self.tol_memory = tol_memory
        self.tol_coef = tol_coef
        self.window_policy = window_policy

    def transform(self, X):
        """
        Return fractional differentiation of array.

        Parameters
        ----------
        - X : array-like, shape (n_samples, n_features)
            Time-series to perform fractional differentiation.

        Returns
        -------
        - X_d : array-like, shape (n_samples, n_features)
            Fractionally differentiated time-series.
            The beginning `self.window - 1` terms will be filled with
            `numpy.nan`.
        """
        X = check_array(X, estimator=self)
        self._fit()

        n_samples, n_features = X.shape

        if n_features > 1:
            return np.hstack([
                self.transform(X[:, :1]),
                self.transform(X[:, 1:]),
            ])

        # Note on implementation:
        #
        # For window = 4, n_samples = 5, n_features = 1,
        #
        # d = [c_3  c_2  c_1  c_0  0.0  0.0  0.0  0.0]
        #      <---- window ---->  <- n_samples - 1 ->
        #
        # D = [
        #     [c_0  0.0  0.0  0.0  0.0  c_3  c_2  c_1] >
        #     [c_1  c_0  0.0  0.0  0.0  0.0  c_3  c_2] |
        #     [c_2  c_1  c_0  0.0  0.0  0.0  0.0  c_3] |
        #     [c_3  c_2  c_1  c_0  0.0  0.0  0.0  0.0] window+ n_samples - 1
        #     [0.0  c_3  c_2  c_1  c_0  0.0  0.0  0.0] |
        #     [0.0  0.0  c_3  c_2  c_1  c_0  0.0  0.0] |
        #     [0.0  0.0  0.0  c_3  c_2  c_0  c_0  0.0] |
        #     [0.0  0.0  0.0  0.0  c_3  c_0  c_0  c_0] >
        # ]    <------ window + n_samples - 1 ------>
        #
        # X_extended
        #   = [
        #     [0.0  0.0  0.0  X_0  X_1  X_2  X_3  X_4]
        #     ]
        #     < window - 1 >  <----- n_samples ----->
        #
        # Y = D @ X_extended
        #   = [...  ...  ...  ...  ...  ...  Y_3  Y_4]
        #                     <--- fracdiff of X --->
        #
        # return
        #   = [               nan  nan  nan  Y_0  Y_1]

        # TODO expanding window method

        d = np.concatenate([self.coef, np.zeros(n_samples - 1)], axis=0)
        D = np.stack([
            np.roll(d, shift - self.window + 1)
            for shift in range(self.window + n_samples - 1)
        ], axis=0)
        X_extended = np.concatenate([np.zeros((self.window - 1, 1)), X], axis=0)

        Xd = (D @ X_extended)[-n_samples:, :]
        Xd[:self.window - 1, :] = np.nan

        return Xd

    def _fit(self):
        """
        Set `self.window` and `self.coef`.
        """
        # TODO Simplify logic
        determine_window = (
            self.window is None
            or getattr(self, '_cached_order', self.order) != self.order
            or getattr(self, '_cached_tol_memory', self.tol_memory) != self.tol_memory
            or getattr(self, '_cached_tol_coef', self.tol_coef) != self.tol_coef
        )
        determine_coef = (
            not hasattr(self, 'coef')
            or getattr(self, '_cached_order', self.order) != self.order
            or getattr(self, '_cached_window', self.window) != self.window
        )
        if not (determine_window or determine_coef):
            return

        if determine_window:
            self.window = None

        if self.window is not None:
            self.coef = self._make_coef(window=self.window)
        elif self.tol_coef is not None:
            self.coef = self._make_coef(tol_coef=self.tol_coef)
            self.window = len(self.coef)
        elif self.tol_memory is not None:
            self.coef = self._make_coef(tol_memory=self.tol_memory)
            self.window = len(self.coef)
        else:
            raise ValueError('None of window, tol_coef, tol_memory are given.')

        self._cached_order = self.order
        self._cached_tol_memory = self.tol_memory
        self._cached_tol_coef = self.tol_coef
        self._cached_window = self.window

    def _make_coef(self, window=None, tol_coef=None, tol_memory=None):
        """
        Generator of coefficients.

        Parameters
        ----------
        - window :
            If specified, truncate up to this number of terms.
        - tol_coef : float in [0, 1]
        - tol_memory : float in [0, 1]

        Note
        ----
        Sequence of coefficients add up to 0.
        """
        window = window or self.MAX_WINDOW
        tol_coef = tol_coef or -np.inf
        tol_memory = tol_memory or -np.inf

        index = 0
        coef = 1.0
        lost_memory = 1.0

        coefs = []

        while (
            index < window
            and abs(coef) > tol_coef
            and lost_memory > tol_memory
        ):
            coefs.append(coef)
            coef *= (index - self.order) / (index + 1)
            index += 1
            lost_memory -= abs(coef)  # What if order < 0 ?

        return np.flip(np.array(coefs))

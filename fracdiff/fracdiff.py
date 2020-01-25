from sklearn.utils.validation import check_array
import numpy as np


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

        d = np.r_[self._coeff, np.zeros(n_samples - 1)]
        D_extended = np.vstack([
            np.roll(d, shift) for shift in range(n_samples)
        ])
        D = D_extended[:, window:]

        X_d = D @ X

        if self.window != -1:
            X_d[:window] = np.nan

        return X_d

    @property
    def _coeff(self):
        # If coeff has been computed with the same params
        if hasattr(self, '_cached_coeff') \
                and self.__order == self.order \
                and self.__window == self.window:
            return self._cached_coeff

        def omega(self):
            c = 1.0
            for k in range(self.window + 1):
                yield c
                c *= (k - self.order) / (k + 1)

        coeff = np.flip(np.array(list(omega(self))))
        self._cached_coeff = coeff
        self.__order = self.order
        self.__window = self.window

        return coeff

    def __check_order(self):
        """Check if the value of order is sane"""
        if not (isinstance(self.order, float) or isinstance(self.order, int)):
            raise ValueError('order must be int or float.')

    def __check_window(self):
        """Check if the value of window is sane"""
        if not isinstance(self.window, int):
            raise ValueError('window must be int.')
        if not (self.window == -1 or self.window > 0):
            raise ValueError('window must be -1 or positive integer.')

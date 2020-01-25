from statsmodels.tsa.stattools import adfuller


class StationarityTester:
    """
    Carry out stationarity test of time-series.

    Parameters
    ----------
    - method : {'ADF'}, default 'ADF'
        'ADF' : Augmented Dickey-Fuller unit-root test.

    Examples
    --------
    >>> np.random.seed(42)
    >>> gauss = np.random.randn(100)  # stationary
    >>> stat = StationarityTester(method='ADF')
    >>> stat.pvalue(gauss)
    1.16e-17
    >>> brown = gauss.cumsum()  # not stationary
    >>> stat.pvalue(brown)
    0.60
    """
    def __init__(self, method='ADF'):
        self.method = method

    @property
    def null_hypothesis(self):
        if self.method == 'ADF':
            return 'unit-root'

    def pvalue(self, x):
        """
        Return p-value of the stationarity test.

        Parameters
        ----------
        - x : array-like, shape (n_samples, )
            Time-series to score p-value.

        Returns
        -------
        pvalue : float
            p-value of the stationarity test.
        """
        if self.method == 'ADF':
            _, pvalue, _, _, _, _ = adfuller(x)
            return pvalue
        raise ValueError(f'Invalid method: {self.method}')

    def is_stationary(self, x, y=None, pvalue=.05):
        """
        Return whether stationarity test implies stationarity.

        Note
        ----
        The name 'is_stationary' may be misleading.
        Strictly speaking, `is_stationary = True` implies that the
        null-hypothesis of the presence of a unit-root has been rejected
        (ADF test) or the null-hypothesis of the absence of a unit-root has
        not been rejected (KPSS test).

        Returns
        -------
        is_stationary : bool
            True may imply the stationarity.
        """
        if self.null_hypothesis == 'unit-root':
            return self.pvalue(x) < pvalue

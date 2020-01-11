from statsmodels.tsa.stattools import adfuller

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

    def pvalue(self, X, y=None, value='pvalue'):
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



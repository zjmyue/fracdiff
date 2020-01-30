from math import factorial
import numpy as np


def _poch(d, n):
    """
    Returns
    -------
    d * (d - 1) * (d - 2) * ... * (d - n + 1) : float
    """
    return np.prod([d - k for k in range(n)])


def get_coefs(d, n_terms):
    """
    Compute coefficients of fracdiff operator explicitly (but inefficiently)
    and return the sequence of them as `numpy.array`.
    Parameters
    ----------
    - d : float
        Order of fracdiff.
    - n_terms : int
        Number of coefficients to compute.
    """
    return np.array([
        (-1) ** k * _poch(d, k) / factorial(k) for k in range(n_terms)
    ])

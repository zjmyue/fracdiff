import pytest


def coeff(d, k):
    def pochhammer(x, n):
        if n == 0:
            return x
        return x * pochhammer(x - 1, n - 1)

    def factorial(x):
        return pochhammer(x, x)

    return (-1) ** k * pochhammer(d, k) // factorial(k)

def test_init(self):
    pass

import pytest
from ._coef import get_coefs

import numpy as np
from fracdiff import Fracdiff


params_d = list(np.linspace(0.0, 3.0, 31))
params_window = [10]
params_n_terms = [10]


# --------------------------------------------------------------------------------


@pytest.mark.parametrize('d', params_d)
@pytest.mark.parametrize('window', params_window)
@pytest.mark.parametrize('n_terms', params_n_terms)
def test_coef(d, window, n_terms):
    coef = Fracdiff(d, window=window)._fit().coef
    coef_expected = get_coefs(d, n_terms)

    assert np.allclose(coef, coef_expected)

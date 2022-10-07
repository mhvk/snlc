import numpy as np
import astropy.units as u
import pytest

from ..arnett import Arnett, deposition, nuclear_a96


class TestArnett:
    @pytest.mark.parametrize('recombination,max_lum', [
        (None, 4.2e41*u.erg/u.s),
        ('fast', 9.41e41*u.erg/u.s),
        ('slow', 12.41e41*u.erg/u.s)])
    def test_it_runs(self, recombination, max_lum):
        sn = Arnett(nuclear=nuclear_a96)
        sol = sn(recombination=recombination)
        assert u.isclose(sol['l'].max(), max_lum, rtol=0.03)



def test_deposition():
    # Check with Table 1 of Colgate et al., 1980, ApJ 237, L81.
    assert np.allclose(deposition(2.**(np.arange(4, -4, -1))),
                       [0.965, 0.930, 0.857, 0.725,
                        0.517, 0.301, 0.158, 0.080], rtol=0.04)

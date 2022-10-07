import numpy as np
import astropy.units as u
import pytest

from ..arnett import Arnett


class TestArnett:
    @pytest.mark.parametrize('recombination,max_lum', [
        (None, 4.2e41*u.erg/u.s),
        ('fast', 9.41e41*u.erg/u.s),
        ('slow', 12.41e41*u.erg/u.s)])
    def test_it_runs(self, recombination, max_lum):
        sn = Arnett()
        sol = sn(recombination=recombination)
        assert u.isclose(sol['l'].max(), max_lum, rtol=0.03)

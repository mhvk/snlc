import numpy as np
import astropy.units as u

from ..arnett import Arnett


class TestArnett:
    def test_it_runs(self):
        sn = Arnett()
        sol = sn()
        assert u.isclose(sol['l'].max(), 4.2e41*u.erg/u.s, rtol=0.03)

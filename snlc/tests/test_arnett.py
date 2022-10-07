import numpy as np
import astropy.units as u
import pytest

from ..arnett import Arnett, deposition, nuclear_a96, nuclear_afm17
from ..figures import SN2011fe


class TestArnett:
    @pytest.mark.parametrize('recombination,max_lum', [
        (None, 4.2e41*u.erg/u.s),
        ('fast', 9.41e41*u.erg/u.s),
        ('slow', 12.41e41*u.erg/u.s)])
    def test_it_runs(self, recombination, max_lum):
        sn = Arnett(nuclear=nuclear_a96)
        sol = sn(recombination=recombination)
        assert u.isclose(sol['l'].max(), max_lum, rtol=0.03)


class TestAFM17:
    def setup_class(self):
        self.sn = Arnett(**SN2011fe, nuclear=nuclear_afm17)
        self.t = np.linspace(0, 40, 401) << u.day
        self.afm17 = self.sn(self.t)

    def test_table2(self):
        """Compare with values in Table 2 of AFM17."""
        for attr, expected in [
                ('vsc', 9.28e8*u.cm/u.s),  # set by hand
                ('te0', 0.43*u.s),  # ~OK
                ('td0', 1.52e12*u.s),  # OK
                ('rho0', 1.04e7*u.g/u.cm**3),  # OK
                # ('T0', 8.77e8*u.K),  # Skip cannot reproduce
                ('tau0', 3.73e14),  # OK
        ]:
            value = getattr(self.sn, attr)
            assert u.isclose(value, expected, rtol=0.05)

    def test_maximum(self):
        """Check maximum luminosity, below Fig. 1."""
        imax = self.afm17['l'].argmax()
        lmax = self.afm17['l'][imax]
        tmax = self.afm17['t'][imax]

        assert u.isclose(tmax, 14.04*u.day, rtol=0.01)
        assert u.isclose(lmax, 3.00e9*u.Lsun, rtol=0.01)


def test_deposition():
    # Check with Table 1 of Colgate et al., 1980, ApJ 237, L81.
    assert np.allclose(deposition(2.**(np.arange(4, -4, -1))),
                       [0.965, 0.930, 0.857, 0.725,
                        0.517, 0.301, 0.158, 0.080], rtol=0.04)

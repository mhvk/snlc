import numpy as np
import astropy.units as u
import pytest

from ..arnett import Arnett, deposition, nuclear_a96, nuclear_afm17
from ..figures import SN2011fe


def test_nuclear():
    """Test total energy generation rates from AFM17"""
    for attr, expected in [
            ('tau_ni', 6.075*u.day/np.log(2)),
            ('egni', 3.9805e10*u.erg/u.g/u.s),
            ('tau_co', 77.236*u.day/np.log(2)),
            ('egco', 6.4552e9*u.erg/u.g/u.s),
            ('epco', 2.1458e8*u.erg/u.g/u.s)]:
        value = getattr(nuclear_afm17, attr)
        assert u.isclose(value, expected, rtol=1*u.percent)


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
        self.sn = Arnett(**SN2011fe)
        self.t = np.linspace(0, 40, 401) << u.day

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

    def test_at_14d14(self):
        """Check properties at maximum from Table 3."""
        t = np.array([0, 14.14, 40]) * u.day
        check = self.sn(t)[1]
        fr = check['fr']
        rho = self.sn.rho0 / fr**3
        # Tc, Teff to be done
        tau = self.sn.tau0 / fr ** 2
        taug = self.sn.tau_gamma0 / fr**2
        d = deposition(taug)

        assert u.isclose(check['fni'], 0.199, rtol=0.01)
        assert u.isclose(check['fco'], 0.740, rtol=0.01)
        assert u.isclose(rho, 4.56e-13*u.g/u.cm**3, rtol=0.015)
        assert u.isclose(tau, 46.6, rtol=0.01)
        # Cannot get taug and deposition(taug) correct at the same time.
        # Sent e-mail to Arnett 2022-10-07.
        # assert u.isclose(taug, 15.5, rtol=0.01)
        assert u.isclose(d, 0.838, rtol=0.01)
        assert u.isclose(check['t'], 14.04*u.day, rtol=0.01)
        assert u.isclose(check['l'], 3.00e9*u.Lsun, rtol=0.015)

    def test_maximum(self):
        check = self.sn(self.t)
        assert u.isclose(check[check['l'].argmax()]['t'], 14.14*u.day,
                         atol=0.5*u.day)


def test_deposition():
    # Check with Table 1 of Colgate et al., 1980, ApJ 237, L81.
    assert np.allclose(deposition(2.**(np.arange(4, -4, -1))),
                       [0.965, 0.930, 0.857, 0.725,
                        0.517, 0.301, 0.158, 0.080], rtol=0.04)

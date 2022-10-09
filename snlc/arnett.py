from dataclasses import dataclass

import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.utils.decorators import lazyproperty
from astropy.table import QTable
from scipy.integrate import solve_ivp


# Nuclear physics data
@dataclass
class NuclearData:
    tni: u.Quantity
    """56Ni half life."""
    gni: u.Quantity
    """56Ni average gamma ray energy."""
    tco: u.Quantity
    """56Co half life."""
    gco: u.Quantity
    """56Co average gamma ray energy."""
    pco: u.Quantity
    """56Co average positron energy (annih. rad. assumed to escape)."""
    kappa_g: u.Quantity = (35.5 * u.g / u.cm**2)**-1
    """Absorption opacity to gamma rays (Colgate et al. 1980, ApJ 237, L81)."""
    name: str = ''

    m56ni = 55.942132 * const.u  # Isotopic masses from Wikipedia
    m56co = 55.9398393 * const.u

    @lazyproperty
    def tau_ni(self):
        return self.tni / np.log(2.)

    @lazyproperty
    def tau_co(self):
        return self.tco / np.log(2.)

    @lazyproperty
    def egni(self):
        """56Ni gamma-ray energy generation rate per unit mass."""
        return (self.gni/self.m56ni/self.tau_ni).to(u.erg/u.g/u.s)

    @lazyproperty
    def egco(self):
        """56Co gamma-ray energy generation rate per unit mass."""

        return (self.gco/self.m56co/self.tau_co).to(u.erg/u.g/u.s)

    @lazyproperty
    def epco(self):
        """56Co positron energy generation rate per unit mass."""
        return (self.pco/(56*const.m_p)/self.tau_co).to(u.erg/u.g/u.s)


nuclear_afm17 = NuclearData(
    name='AFM17',
    tni=6.075*u.day,
    gni=1.750*u.MeV,
    tco=77.236*u.day,
    gco=3.610*u.MeV,
    pco=0.120*u.MeV,
    kappa_g=0.00716*u.cm**2/u.g,
    # Empirical override, to get D(tau_g(max)) right.
    # Alternative, to get tau_g(max) right.
    # kappa_g=0.03*u.cm**2/u.g,
    # (Close but not quite right with mfp=35.5g/cm^2 from Colgate et al.)
)
"""Nuclear data listed in Arnett, Fryer & Matheson 2017"""


nuclear_a96 = NuclearData(
    name='A96',
    tni=6.10*u.day,
    gni=1.72*u.MeV,
    tco=77.12*u.day,
    # Cobalt energy input, gammas and positrons seperately (Arnett, Fig. 13.4)
    gco=(0.279*4.10
         + 0.167*3.856
         + 0.219*3.445
         + (0.078+0.009)*3.123
         + (0.024+0.181)*2.085
         ) * u.MeV,
    pco=(0.5*(0.009*(4.556-3.123)
              + 0.181*(4.556-2.085))
         + (0.009+0.181)*0.512*2
         ) * u.MeV
)
"""Nuclear data listed in Arnett 1996 (book)"""


def deposition(tau):
    """Deposition rate for given optical depth in gamma rays.

    From Colgate et al., 1980, ApJ 237, L81
    """
    g = tau / (1.6+tau)
    return g * (1 + 2*g*(1-g)*(1-0.75*g))


class Arnett:
    """Calulate Arnett-stylie lightcurve.

    Defaults are SN 1987A parameters from Arnett's Table 13.2.

    Parameters
    ----------
    esn: |Quantity|
        Supernova energy.
    mej: |Quantity|
        Ejecta mass.
    r0 : |Quantity|
        Initial radius.
    kappa: |Quantity|
        Opacity per unit mass.
    mni: |Quantity|
        Nickel mass.
    tion: |Quantity|
        Recombination temperature.
    qion: |Quantity|
        Recombination energy per unit mass.
    vsc : |Quantity|, optional
        Scale velocity.  If not given, calculated from the supernova energy
        using ``sqrt(esn/mej*5/3)`` (A96), but in other papers it is estimated
        differently.
    beta : float
        Diffusion constant, appropriate for constant density.
    nuclear : NuclearData, optional
        Set of nuclear data to use.  By default, those of A96.
    """

    def __init__(self,
                 *,
                 esn=1.7e51*u.erg,
                 mej=15*u.Msun,
                 r0=3.0e12*u.cm,
                 kappa=0.2*u.cm**2/u.g,
                 mni=0.075*u.Msun,
                 tion=4500*u.K,
                 qion=0.7*13.6*u.eV/const.m_p,
                 vsc=None,
                 beta=13.8,
                 nuclear=nuclear_a96,
                 ):
        self.mej = mej
        self.esn = esn
        self.r0 = r0
        self.kappa = kappa
        self.mni = mni
        self.tion = tion
        self.qion = qion
        self.beta = beta
        self.nuclear = nuclear
        # Expansion velocity of outermost layer (acceleration already applied),
        # or the scale velocity (eq. D33).
        self.vsc = (vsc if vsc is not None
                    else ((2*esn/2/mej*5./3.)**0.5)).to(u.km/u.s)
        self.te0 = r0/self.vsc
        # Initial thermal energy [erg] (half SN energy, rest kinetic)
        # (use Etot=Esn/2 by comparison with Rph as a fu. of time in Fig.13.11)
        self.eth0 = esn/2
        # Initial diffusion time scale.
        self.td0 = (kappa*mej/self.beta/const.c/r0).to(u.day)
        # Mean density.
        volume = 4*np.pi/3.*r0**3
        self.rho0 = (mej/volume).to(u.g/u.cm**3)
        # Initial dimensionless location of photosphere.
        # Calculate initial location of photosphere (not really important, but
        # for some conditions, it is better that xi<1).
        self.xi0 = 1 - 1/(kappa*self.rho0)/r0
        # For reference.
        self.tau0 = self.rho0 * r0 * kappa
        self.T0 = ((self.eth0 / volume)
                   / (4 * const.sigma_sb / const.c)) ** 0.25
        # --- radioactive heating
        # Heating timescale (=1/p1 of AF89).
        self.th0 = (self.eth0 / (self.nuclear.egni*mni)
                    if mni > 0 else np.inf << u.s)
        # --> TODO: how is this actually used???
        # Fractional radius where Ni is located (for escape).
        self.xni = 0.65
        self.tau_gamma0 = r0 * nuclear.kappa_g * self.rho0
        # -- Recombination
        # Radiation time at Teff=2**(1/4)Tion.
        self.ti0 = (self.eth0 / (4*np.pi*r0**2 * const.sigma_sb*2*tion**4)
                    ).to(u.day)
        # Ratio of recombination to thermal energy.
        self.feion0 = (mej*qion / self.eth0).to(u.one)

    def __call__(self,
                 times=np.arange(0, 400., 1.)*u.day,
                 *,
                 recombination=None):
        """Calculate the lightcurve.

        Parameters
        ----------
        times: |Quantity|
            Times at which to evaluate the lightcurve.
        recombination: {'fast', 'slow'}, None
            Whether recombination is treated, and if so, whether it is
            fast or slow relative to diffusion.

        Returns
        -------
        result : |QTable|
            With columns ``t`` (time), ``phi``, ``xi_raw``, ``fni``, ``fco``
            (dimensionless energy, ionization front radius, fraction of 56Ni,
            fraction of 56Co), ``xi`` (negative values removed), and
            ``r``, ``l``, ``teff`` (radius, luminosity, and effective temp.).
        """

        self._time_unit = times.unit
        # --> evolve it!  Note that solve_ivp cannot handle Quantities, so we
        # use arrays in whatever unit the times are in.
        # y0 containts the initial values for phi, xi, fni, and fco.
        start = dict(fr=1, fni=1, fco=0, phi=1)
        if recombination is None:
            derivatives = self.base

        else:
            derivatives = getattr(self, recombination)
            start['xi'] = self.xi0.value

        sol = solve_ivp(derivatives,
                        t_span=times[[0, -1]].value,
                        y0=list(start.values()),
                        t_eval=times.value)
        # TODO: check no errors occurred!
        result = QTable([times, *sol.y], names=['t']+list(start.keys()))
        # --> calculate physical properties.
        phi = result['phi']
        xi = result['xi'] = (1 if recombination is None
                             else np.maximum(result['xi'], 0.))
        # Photospheric radius.
        r = result['r'] = result['fr'] * self.r0 * xi
        # TODO: should be a method that can be called? Can it be generalized?
        ldiff = (phi * self.eth0 / self.td0).to(u.erg/u.s)
        if recombination is not None:
            lmin = (r/self.r0)**2 * self.eth0 / self.ti0
            if recombination == "fast":
                # Rescale to cut-off temperature distribution.
                ldiff = ldiff * (-xi * np.cos(np.pi*xi)
                                 + np.sin(np.pi*xi) / np.pi)
            elif recombination == "slow":
                # Rescale to normal temperature distribution
                # but at smaller radius.
                ldiff = ldiff * xi

            ldiff = np.maximum(ldiff, lmin)

        result['l'] = ldiff
        result['teff'] = ((result['l'] / (4*np.pi*r**2)
                           / const.sigma_sb)**(1/4)).to(u.K)
        return result

    def heating(self, t, fr, fni, fco):
        """Calculate Ni and Co decay and associated heating."""
        fni_dot = -fni / self.nuclear.tau_ni  # 56Ni decay rate.
        fco_dot = (fni / self.nuclear.tau_ni
                   - fco / self.nuclear.tau_co)  # 56Co creation and decay.
        # εM/E, normalised to initial heating (zeta(t) of AF89).
        tau_gamma = self.tau_gamma0 / fr**2
        depfrac = deposition(tau_gamma)
        heating = (fni * depfrac
                   + ((self.nuclear.egco * depfrac + self.nuclear.epco)
                      / self.nuclear.egni * fco))

        # Associated ϕ-dot, corrected for adiabatic expansion.
        phi_dot = heating / self.th0 * fr
        return fni_dot, fco_dot, phi_dot

    def base(self, t, par):
        t = t << self._time_unit
        fr, fni, fco, phi = par
        # Calculate composition and energy changes due to radioactive decay.
        fni_dot, fco_dot, phi_dot = self.heating(t, fr, fni, fco)
        # Subtract cooling by diffusion.
        phi_dot -= phi/self.td0 * fr
        return u.Quantity([1/self.te0, fni_dot, fco_dot, phi_dot],
                          unit=1/self._time_unit).value

    def fast(self, t, par):
        t = t << self._time_unit
        fr, fni, fco, phi, xi = par
        xi2dpsidxi = -xi * np.cos(np.pi*xi) + np.sin(np.pi*xi) / np.pi
        volkept = xi2dpsidxi
        pi2by3psi = np.pi**2 / 3 * np.sin(np.pi*xi)/(np.pi*xi)
        # Calculate dphi/dt and dxi/dt.
        # Effectively, the first term applies all heating in the centre.
        fni_dot, fco_dot, phi_dot = self.heating(t, fr, fni, fco)
        phi_dot /= volkept
        # Subtract cooling.
        phi_dot -= phi / self.td0 * fr
        # xidot = (<emitted>-<diffused>)/(<advected>+<recombined>)
        xi_dot = np.minimum(  # Recombination front can only go in.
            ((xi*fr)**2 / self.ti0
             - xi2dpsidxi*phi / self.td0)
            / (self.feion0 + phi*pi2by3psi/fr)/-3/xi**2,
            0.)

        if getattr(self, 'debug', False) and abs(t-100*u.day) < 5*u.day:
            # Sanity check of luminosities: lmin2 should equal lmin.
            lmin2 = (4*np.pi*(self.r0*fr*xi)**2
                     * const.sigma_sb * (2.*self.tion**4))
            ldiff = self.eth0 * phi / self.td0 * xi2dpsidxi
            ladv = -3*xi**2 * xi_dot * pi2by3psi*xi * self.eth0 * phi / fr
            lion = -3. * xi**2 * xi_dot * self.feion0 * self.eth0
            lmin = (xi*fr)**2 * self.eth0 / self.ti0
            # Reget heating
            _, _, heating = self.heating(t, fni, fco)
            lheat = (heating / fr) * self.eth0
            fmt = "{0}={1:10.4e}"
            vals = locals().copy()
            print(f"{t=:5.1f}, {phi=:5.3f}, {xi=:5.3f}, " + ', '.join([
                fmt.format(x, vals[x].to_value(u.erg/u.s))
                for x in ('ldiff', 'ladv', 'lion', 'lmin', 'lheat')]))
            print(f"{t=:5.1f}, {phi=:5.3f}, {xi=:5.3f}, "
                  f"phi_dot={phi_dot.to(1/u.day):10.4g}, "
                  f"xi_dot={xi_dot.to(1/u.day):10.4g}, "
                  + fmt.format('lmin2', lmin2.to_value(u.erg/u.s)))

        return u.Quantity([1/self.te0, fni_dot, fco_dot, phi_dot, xi_dot],
                          unit=1/self._time_unit).value

    def slow(self, t, par):
        # Have inconsistencies with what is shown in Arnett.
        # Changing phi(2) to fractional volume left does not matter;
        # solving ln(phi), ln(xi) does not matter.
        t = t << self._time_unit
        # TODO: factor out; is the same as for fast().
        fr, fni, fco, phi, xi = par
        fni_dot, fco_dot, phi_dot = self.heating(t, fr, fni, fco)
        # AF89: phidot = sigma/xi**3[p1*zeta -p2*phi  -2(phi/sigma)xi**2 xidot]
        #              = fr/xi**3 [ zeta/th0 -phi/td0 -2(phi/fr)xi**2 xidot]
        # Effectively, this applies all heating in the centre
        # Arnett book, eq. 13.39
        # dϕ/dt = [(Mε+Lpsr)/E₀xᵢ³ - ϕ/xᵢ²τ₀](R/R₀)
        # Adjust heating for changed volume.
        phi_dot /= xi**3
        if xi > 0.1:
            # Subtract cooling.
            phi_dot -= phi/xi**2/self.td0 * fr
            # Arnett book, eq. 13.40
            # 3xᵢ²dxᵢ/dt = [ϕ/τ₀ - (π²c/3R₀)(Tᵢ⁴/2T₀⁴)(R/R₀)⁴]
            #              / [MQᵢ/E₀ + ϕ(R₀/R)]
            # I defined
            # tᵢ₀ = E₀ / Lᵢ₀
            #     = (4/π)R₀³aT₀⁴ / 4πR₀²(ac/4)2Tᵢ⁴ = (4R₀/π²c)(T₀⁴/2Tᵢ⁴)
            # I assume there are typos in Arnett's 13.40
            # (3 instead of 4, 2 in front of T₀ instead of Tᵢ)
            # Derivation for me:
            # Lmin = Lrec+Ladv+Ldiff = 4πRᵢ²σ2Tᵢ⁴ = xᵢ²(R/R₀)² 4πR₀²σ2Tᵢ⁴
            #      = xᵢ²(R/R₀)²(Lᵢ/E₀)E₀
            # Lrec = 4πRᵢ²dRᵢ/dt ρQ = -3xᵢ²dxᵢ/dt (4π/3)R³ρQ = -3xᵢ²dxᵢ/dt MQ
            #      = -3xᵢ²dxᵢ/dt (MQ/E₀)E₀
            # Ladv = -dxᵢ/dt ∂E/∂xᵢ = -3xᵢ²dxᵢ/dt E₀(R₀/R)ϕ
            # Ldiff = E/τ_d = E₀(R₀/R)ϕxᵢ³/τ₀(R₀/R)xᵢ² = E₀ϕxᵢ/τ₀
            # Hence, -3xᵢ²dxᵢ/dt = (xᵢ²(R/R₀)²/tᵢ₀-ϕ/τ₀) / (MQ/E₀ + ϕ(R₀/R))
            xi_dot = np.minimum(  # minimum since Teff may be larger than ⁴√2Tᵢ
                ((xi*fr)**2/self.ti0 - phi*xi/self.td0)
                / (self.feion0 + phi/fr)
                / (-3*xi**2),
                0.)
        else:
            # TODO: is any of this correct???
            # With far-advanced recombination radius, really we're just
            # matching the input from radioactive heating with the luminosity,
            # so we need heating*eth0/th0=xi**2*fr**2*eth0/ti0
            # hence, dln h = 2 dln xi + 2d ln fr
            #    -> dln xi = 0.5* dln h -vsc/r0/fr
            heating = phi_dot * self.th0 / fr
            xi_dot = xi*(0.5*(fni_dot
                              + ((self.nuclear.egco+self.nuclear.epco)
                                 / self.nuclear.egni*fco_dot)/heating)
                         - self.vsc / (self.r0+self.vsc*t))
            # Part below unclear whether this is correct.
            # this implies Ldiff=Lmin-Ladv-Lrec=heating-Ladv-Lrec
            #    phi_dot = (heating/xi**3/self.th0
            #               - (xi*fr)**2/xi**3/self.ti0
            #               - 3*xi_dot/xi*(self.feion0/phi+1/fr))*fr
            #    phi_dot = (heating/xi**3/self.th0
            #               - (xi*fr)**2/xi**3/self.ti0)*fr
            # print(f"heat={heating/xi**3/self.th0}, "
            #       f"min={(xi*fr)**2/xi**3/self.ti0, "
            #       f"rec={-3*xi_dot/xi*self.feion0/phi}, "
            #       f"adv={-3*xi_dot/xi/fr}")
            # Since heating now equals cooling, no change in phi.
            phi_dot = 0./u.day

        if getattr(self, 'debug', False) and abs(t-100*u.day) < 5*u.day:
            # Sanity check of luminosities: lmin2 should equal lmin.
            lmin2 = (4*np.pi*(self.r0*fr*xi)**2
                     * const.sigma_sb * (2.*self.tion**4))
            ldiff = self.eth0 * phi / self.td0 * xi
            ladv = -3. * xi**2 * xi_dot * self.eth0 * phi / fr
            lion = -3. * xi**2 * xi_dot * self.feion0 * self.eth0
            lmin = (xi*fr)**2 * self.eth0 / self.ti0
            # Reget heating
            _, _, heating = self.heating(t, fni, fco)
            lheat = (heating / fr) * self.eth0
            fmt = "{0}={1:10.4e}"
            vals = locals().copy()
            print(f"{t=:5.1f}, {phi=:5.3f}, {xi=:5.3f}, " + ', '.join([
                fmt.format(x, vals[x].to_value(u.erg/u.s))
                for x in ('ldiff', 'ladv', 'lion', 'lmin', 'lheat')]))
            print(f"{t=:5.1f}, {phi=:5.3f}, {xi=:5.3f}, "
                  f"phi_dot={phi_dot.to(1/u.day):10.4g}, "
                  f"xi_dot={xi_dot.to(1/u.day):10.4g}, "
                  + fmt.format('lmin2', lmin2.to_value(u.erg/u.s)))

        return u.Quantity([1/self.te0, fni_dot, fco_dot, phi_dot, xi_dot],
                          unit=1/self._time_unit).value


if __name__ == '__main__':
    arnett = Arnett()
    norec = arnett()
    fast = arnett(recombination='fast')
    slow = arnett(recombination='slow')

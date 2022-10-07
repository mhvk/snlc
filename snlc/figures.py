import numpy as np
import astropy.units as u
import astropy.constants as const
import matplotlib.pylab as plt

from .arnett import Arnett, nuclear_afm17


SN1987A = dict(
    esn=1.7e51*u.erg,
    mej=15*u.Msun,
    r0=3.0e12*u.cm,
    kappa=0.2*u.cm**2/u.g,
    mni=0.075*u.Msun,
    tion=4500*u.K,
    qion=0.7*13.6*u.eV/const.m_p
    )


SN2011fe = dict(  # From Arnett, Fryer & Matheson, 2017, ApJ 846:33
    mej=1.4*u.Msun,
    mni=0.546*u.Msun,
    r0=4e8*u.cm,
    kappa=0.09*u.cm**2/u.g,
    esn=1.2e51*u.erg,
    vsc=9.28e8*u.cm/u.s,  # Override default calculation from esn.
    )


def comparison(recombination='fast'):
    t = np.linspace(0, 350, 351) << u.day
    base = Arnett(**SN1987A)
    full = base(t, recombination=recombination)
    norec = base(t, recombination=None)
    nonickel = Arnett(**(SN1987A | dict(mni=0*u.Msun)))
    noheat = nonickel(t, recombination=recombination)
    diff = nonickel(t, recombination=None)
    plt.plot(t, full['l'], label='heating, recombination')
    plt.plot(t, norec['l'], label='heating')
    plt.plot(t, noheat['l'], label='recombination')
    plt.plot(t, diff['l'], label='diffusion only')
    plt.yscale('log')
    plt.xlim(0, 350)
    plt.ylim(1e40, 2e42)
    plt.legend()


def exploration(recombination='fast'):
    t = np.linspace(0, 350, 351) << u.day
    var = dict(
        r0=10.**np.linspace(12, 14, 5) << u.cm,
        esn=10.**np.linspace(50.5, 52.5, 5) << u.erg,
        mej=10.**np.linspace(0, 1.5, 4) << u.Msun,
        mni=10.**np.linspace(-2, 0.5, 6) << u.Msun)
    for j, (item, values) in enumerate(var.items()):
        plt.subplot(2, 2, j+1)
        plt.title(f"{item} ({values.unit.to_string('latex')})")
        for val in values:
            model = Arnett(**(SN1987A | {item: val}))(t, recombination=recombination)
            plt.plot(t, model['l'], label=f"{val.value:<.2g}")
        plt.yscale('log')
        plt.xlim(0, 350)
        plt.ylim(1e40, 2e43)
        plt.legend()


def sn2011fe():
    t = np.linspace(0, 40, 41) << u.day
    base = Arnett(**SN2011fe)
    default = base(t)
    plt.plot(t, default['l'].to(u.Lsun), label='AFM17')
    return base


if __name__ == '__main__':
    # comparison()
    # plt.savefig('comparison.png')
    # exploration()
    # plt.savefig('exploration.png')
    pass

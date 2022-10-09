****************************
Semi-analytic SN lightcurves
****************************

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy

.. image:: https://github.com/mhvk/snlc/workflows/CI/badge.svg
    :target: https://github.com/mhvk/snlc/actions
    :alt: Test Status

``snlc`` calculates semi-analytic supernova lightcurves using Arnett's
approximation that the spatial and temporal structure can be
separated.  It relies on `NumPy <http://www.numpy.org/>`_, `Astropy
<http://www.astropy.org/>`_, and `matplotlib <https://matplotlib.org/>`_.

It was created for a University of Toronto transients class, and
likely contains bugs.

.. Installation

Installation instructions
=========================

The package and its dependencies can be installed with::

  pip install git+https://github.com/mhvk/snlc.git#egg=snlc

Basic example
=============

The following is to get a quick lightcurve with parameters like those
for SN 2011fe.

    >>> import numpy as np
    >>> import astropy.units as u
    >>> import matplotlib.pyplot as plt
    >>> from snlc.arnett import Arnett
    >>> from snlc.figures import SN2011fe
    >>> model = Arnett(**SN2011fe)
    >>> t = np.linspace(0, 40, 41) << u.day
    >>> lc = model(t)
    >>> plt.plot(t, lc['l'].to(u.Lsun), label='AFM17')

Do inspect the ``SN2011fe`` dict with parameters, and the output
``lc``!  The ``figures.py`` file and the tests are useful places to
get started.

Contributing
============

Please open a new issue for bugs, feedback or feature requests.

We welcome code contributions, including pieces of code to reproduce
similar efforts in the literature.  To add a contribution, please
submit a pull request.  If you would like assistance, please feel free
to contact `@mhvk`_.

For more information on how to make code contributions, please see the `Astropy
developer documentation <http://docs.astropy.org/en/stable/index.html#developer-documentation)>`_.

License
=======

``snlc`` is licensed under the GNU General Public License v3.0 - see the
``LICENSE`` file.

.. _@mhvk: https://github.com/mhvk

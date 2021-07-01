======
MATSgp
======


.. image:: https://img.shields.io/pypi/v/MATSgp.svg
        :target: https://pypi.python.org/pypi/MATSgp

.. image:: https://img.shields.io/travis/wpk-nist-gov/MATSgp.svg
        :target: https://travis-ci.com/wpk-nist-gov/MATSgp

.. image:: https://readthedocs.org/projects/MATSgp/badge/?version=latest
        :target: https://MATSgp.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




Gaussian process regression extension for MATS


* Free software: NIST license
* Documentation: https://MATSgp.readthedocs.io.

Bare-bones install
------------------

To add pre-commit hooks do

.. code-block:: console

    $ make pre-commit-init

Note that because of the pip dependencies, I've but everything for development in environment-dev.yml (instead of the main stuff in environment.yml, and only the add ons in environment-dev.yml).  To make the environment, do


.. code-block:: console

    $ make conda-dev

Or, if you have mamba installed


.. code-block:: console

    $ make mamba-dev


This will install everything, including MATS.  Note that the MATS is based off of https://github.com/wpk-nist-gov/MATS/tree/cci-from-Development/MATS


To install MATSgp, do the following:

.. code-block:: console

   $ make install-dev







Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `wpk-nist-gov/cookiecutter-pypackage`_ Project template forked from `audreyr/cookiecutter-pypackage`_.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`wpk-nist-gov/cookiecutter-pypackage`: https://github.com/wpk-nist-gov/cookiecutter-pypackage
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

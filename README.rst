========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - tests
      - | |travis| |appveyor| |requires|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|

.. |travis| image:: https://api.travis-ci.org/jalammar/ecco.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/jalammar/ecco

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/jalammar/ecco?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/jalammar/ecco

.. |requires| image:: https://requires.io/github/jalammar/ecco/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/jalammar/ecco/requirements/?branch=master

.. |codecov| image:: https://codecov.io/gh/jalammar/ecco/branch/master/graphs/badge.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/jalammar/ecco

.. |version| image:: https://img.shields.io/pypi/v/ecco.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/ecco

.. |wheel| image:: https://img.shields.io/pypi/wheel/ecco.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/ecco

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/ecco.svg
    :alt: Supported versions
    :target: https://pypi.org/project/ecco

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/ecco.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/ecco

.. |commits-since| image:: https://img.shields.io/github/commits-since/jalammar/ecco/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/jalammar/ecco/compare/v0.0.0...master



.. end-badges

Visualization tools for NLP machine learning models.

* Free software: BSD 3-Clause License

Installation
============

::

    pip install ecco

You can also install the in-development version with::

    pip install https://github.com/jalammar/ecco/archive/master.zip


Documentation
=============


To use the project:

.. code-block:: python

    import ecco
    ecco.longest()


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox

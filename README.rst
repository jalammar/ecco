========
Overview
========

.. start-badges

|version| |supported-versions|

.. |version| image:: https://img.shields.io/pypi/v/ecco.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/ecco

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/ecco.svg
    :alt: Supported versions
    :target: https://pypi.org/project/ecco
.. end-badges

[Work in progress. Not yet ready for public consumption]

Visualization tools for NLP machine learning models.

* Free software: BSD 3-Clause License

Installation
============

::

    pip install ecco


Documentation
=============


To use the project:

.. code-block:: python

    import ecco
    lm = ecco.from_pretrained('distilgpt2')
    text= "The countries of the European Union are:\n1. Austria\n2. Belgium\n3. Bulgaria\n4."

    output = lm.generate(text, generate=20, do_sample=True)

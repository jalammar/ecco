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


Ecco
================================
Ecco is a python library for creating interactive visualizations for Natural Language Processing models.

It provides multiple interfaces to aid the explanation and intuition of `Transformer
<https://example.com/>`_-based language models.

Ecco runs inside Jupyter notebooks. It is built on top of `pytorch
<https://pytorch.org/>`_ and `transformers
<https://github.com/huggingface/transformers>`_.


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

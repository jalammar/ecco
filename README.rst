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


Gallery
------------
.. raw:: html

   <div class="container gallery">
      <div class="row">

         <div class="col-lg-6 col-md-6 col-sm-6">

            <strong>Input and Output Sequences</strong>
            <a target="_blank" href="">
            <img src="docs/_static/input-output.PNG" style="border: 1px solid #ddd" />
            </a>
         View the inputs and outputs (generated text) of language models broken down into tokens.

         </div>

         <div class="col-lg-6 col-md-6 col-sm-6">

            <strong>Input Saliency</strong>
            <a target="_blank" href="">
            <img src="docs/_static/input-saliency.PNG" />
            </a>

            <p>How much did each input token contribute to producing the output token?</p>
         </div>


         <div class="col-lg-6 col-md-6 col-sm-6">
            <strong>Neuron factors</strong>
            <a target="_blank" href="">
            <img src="docs/_static/activation-factors.PNG" />
            </a>
            <p>Extracting underlying behaviour of neurons in a small number of factors with dimensionality reduction.</p>
         </div>




      </div>
   </div>

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

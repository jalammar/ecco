.. Ecco documentation master file, created by
   sphinx-quickstart on Sat Nov 28 09:47:43 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

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
            <img src="_static/input-output.PNG" style="border: 1px solid #ddd" />
            </a>
         View the inputs and outputs (generated text) of language models broken down into tokens.

         </div>

         <div class="col-lg-6 col-md-6 col-sm-6">

            <strong>Input Saliency</strong>
            <a target="_blank" href="">
            <img src="_static/input-saliency.PNG" />
            </a>

            <p>How much did each input token contribute to producing the output token?</p>
         </div>



         <div class="col-lg-6 col-md-6 col-sm-6">

            <strong>Predicted Tokens</strong>
            <a target="_blank" href="">
            <img src="_static/layer_predictions.PNG" />
            </a>
            <p>View the model's prediction for the next token (with probability scores). See how the predictions evolved through the model's layers.</p>
         </div>



         <div class="col-lg-6 col-md-6 col-sm-6">
            <strong>Hidden State Evolution</strong>
            <a target="_blank" href="">
            <img src="_static/rankings.PNG" />
            </a>
            <p>After the model picks an output token, Look back at how each layer ranked that token.</p>
         </div>



         <div class="col-lg-6 col-md-6 col-sm-6">
            <strong>Hidden State Evolution - Watch</strong>
            <a target="_blank" href="">
            <img src="_static/rankings_watch.PNG" />
            </a>
            <p>Compare the rankings of multiple tokens as candidates for a certain position in the sequence.</p>
         </div>


         <div class="col-lg-6 col-md-6 col-sm-6">
            <strong>Neuron factors</strong>
            <a target="_blank" href="">
            <img src="_static/activation-factors.PNG" />
            </a>
            <p>Extracting underlying behaviour of neurons in a small number of factors with dimensionality reduction.</p>
         </div>



         <div class="col-lg-6 col-md-6 col-sm-6">
            <strong>Attention</strong>
            <a target="_blank" href="">
            <img src="https://via.placeholder.com/400x200" />
            </a>
            <p>Where did the model pay attention when it was generating/processing each token?</p>
         </div>



         <div class="col-lg-6 col-md-6 col-sm-6">
            <strong>Attention Flow</strong>
            <a target="_blank" href="">
            <img src="https://via.placeholder.com/400x200" />
            </a>
            <p>A more complete view of attention which incorporates residual connections and how previous layers mixed the data from various tokens.</p>
         </div>


      </div>
   </div>



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

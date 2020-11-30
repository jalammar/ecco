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

   <div class="container">
      <div class="row">

         <div class="col-lg-6 col-md-6 col-sm-6">
            <a target="_blank" href="">
            <img src="https://via.placeholder.com/400x200" />
            </a>
            <p>Input and output sequences</p>
         </div>

         <div class="col-lg-6 col-md-6 col-sm-6">
            <a target="_blank" href="">
            <img src="https://via.placeholder.com/400x200" />
            </a>
            <p>Input Saliency</p>

            <p>How much did each input token contribute to producing the output token?</p>
         </div>



         <div class="col-lg-6 col-md-6 col-sm-6">
            <a target="_blank" href="">
            <img src="https://via.placeholder.com/400x200" />
            </a>
            <p>Predicted Tokens</p>
            <p>The model prediction for the next token (with probability scores).</p>
         </div>



         <div class="col-lg-6 col-md-6 col-sm-6">
            <a target="_blank" href="">
            <img src="https://via.placeholder.com/400x200" />
            </a>
            <p>Hidden State Evolution</p>
            <p>After the model picks a token, let's look back at how each layer ranked the output token.</p>
         </div>



         <div class="col-lg-6 col-md-6 col-sm-6">
            <a target="_blank" href="">
            <img src="https://via.placeholder.com/400x200" />
            </a>
            <p>Hidden State Evolution - watched tokens</p>
            <p>Compare the rankings of multiple tokens as candidates for a certain position in the sequence.</p>
         </div>



         <div class="col-lg-6 col-md-6 col-sm-6">
            <a target="_blank" href="">
            <img src="https://via.placeholder.com/400x200" />
            </a>
            <p>Neuron activations</p>
            <p>View the firing patterns of neurons as the model generates output tokens.<p>
         </div>




         <div class="col-lg-6 col-md-6 col-sm-6">
            <a target="_blank" href="">
            <img src="https://via.placeholder.com/400x200" />
            </a>
            <p>Neuron factors</p>
            <p>Extracting underlying behaviour of neurons in a small number of factors with dimensionality reduction.</p>
         </div>



         <div class="col-lg-6 col-md-6 col-sm-6">
            <a target="_blank" href="">
            <img src="https://via.placeholder.com/400x200" />
            </a>
            <p>Attention</p>
            <p>Where did the model pay attention when it was generating/processing each token?</p>
         </div>



         <div class="col-lg-6 col-md-6 col-sm-6">
            <a target="_blank" href="">
            <img src="https://via.placeholder.com/400x200" />
            </a>
            <p>Attention Flow</p>
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

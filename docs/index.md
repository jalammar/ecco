# Welcome to Ecco
Ecco is a python library for explaining Natural Language Processing models using interactive visualizations.

Language models are some of the most fascinating technologies. They are programs that can speak and <i>understand</i> language better than any technology we've had before. For the general audience, Ecco provides an easy way to start interacting with language models. For people closer to NLP, Ecco provides methods to visualize and interact with underlying mechanics of the language models.

Ecco runs inside Jupyter notebooks. It is built on top of [pytorch](https://pytorch.org/) and [transformers](https://github.com/huggingface/transformers).

Ecco is not concerned with training or fine-tuning models. Only exploring and understanding existing pre-trained models.

## Tutorials
- Video: [Take A Look Inside Language Models With Ecco](https://www.youtube.com/watch?v=rHrItfNeuh0)

## How-to Guides
- [Interfaces for Explaining Transformer Language Models](https://jalammar.github.io/explaining-transformers/)
- [Finding the Words to Say: Hidden State Visualizations for Language Models](https://jalammar.github.io/hidden-states/)


## API Reference
The [API reference](api/ecco) and the [architecture](architecture) page explain Ecco's components and how they work together.

## Gallery

<div class="container gallery" markdown="1">

<p><strong>Predicted Tokens:</strong> View the model's prediction for the next token (with probability scores). See how the predictions evolved through the model's layers. [<a href="">Notebook</a>] [<a href="">Colab</a>]</p>
<img src="img/layer_predictions_ex_london.png" />

<p><strong>Rankings across layers:</strong> After the model picks an output token, Look back at how each layer ranked that token.  [<a href="">Notebook</a>] [<a href="">Colab</a>]</p>
<img src="img/rankings_ex_eu_1_widethumb.png" />

<p><strong>Layer Predictions:</strong>Compare the rankings of multiple tokens as candidates for a certain position in the sequence.  [<a href="">Notebook</a>] [<a href="">Colab</a>]</p>
<img src="img/rankings_watch_ex_is_are_widethumb.png" />
<br />
<p><strong>Input Saliency:</strong> How much did each input token contribute to producing the output token?  [<a href="">Notebook</a>] [<a href="">Colab</a>]
</p>
<img src="img/saliency_ex_1_thumbwide.png" />

<p><strong>Detailed Saliency:</strong> See more precise input saliency values using the detailed view. [<a href="">Notebook</a>] [<a href="">Colab</a>]
</p>
<img src="img/saliency_ex_2_thumbwide.png" />

<p><strong>Neuron Activation Analysis:</strong> Examine underlying patterns in neuron activations using non-negative matrix factorization. [<a href="">Notebook</a>] [<a href="">Colab</a>]</p>
<img src="img/nmf_ex_1_widethumb.png" />

</div>

## Getting Help
Having trouble?

- The [Discussion](https://github.com/jalammar/ecco/discussions) board might have some relevant information. If not, you can post your questions there.
- Report bugs at Ecco's [issue tracker](https://github.com/jalammar/ecco/issues)
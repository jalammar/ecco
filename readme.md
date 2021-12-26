![Ecco Logo](https://ar.pegg.io/img/ecco-logo-w-800.png)

[![PyPI Package latest release](https://img.shields.io/pypi/v/ecco.svg)](https://pypi.org/project/ecco)
[![Supported versions](https://img.shields.io/pypi/pyversions/ecco.svg)](https://pypi.org/project/ecco)


Ecco is a python library for exploring and explaining Natural Language Processing models using interactive visualizations. 

Examples:

What is the sentiment of this review?

Which words in this review lead the model to classify its sentiment as "negative"?
[image]
[small caption: Feature attribution using Integrated Gradients on the T5 language model]

Does GPT2 know where Heathrow Airport is?
[image]
[caption: ]

Which input words lead it to think of London?
[image]
[caption:]

What other cities/words did the model consider in addition to London?
[image]
[caption:]

At which layers did the model gather confidence that London is the right answer?
[image]
[caption:]



Read the paper: 
>[Ecco: An Open Source Library for the Explainability of Transformer Language Models](https://aclanthology.org/2021.acl-demo.30/)
> Association for Computational Linguistics (ACL) System Demonstrations, 2021



Ecco provides multiple interfaces to aid the explanation and intuition of [Transformer](https://jalammar.github.io/illustrated-transformer/)-based language models. Read: [Interfaces for Explaining Transformer Language Models](https://jalammar.github.io/explaining-transformers/).

Ecco runs inside Jupyter notebooks. It is built on top of [pytorch](https://pytorch.org/) and [transformers](https://github.com/huggingface/transformers).


Ecco is not concerned with training or fine-tuning models. Only exploring and understanding existing pre-trained models. The library is currently an alpha release of a research project. Not production ready. You're welcome to contribute to make it better!


Documentation: [ecco.readthedocs.io](https://ecco.readthedocs.io/)


## Tutorials
- Video: [Take A Look Inside Language Models With Ecco](https://www.youtube.com/watch?v=rHrItfNeuh0). \[<a href="https://colab.research.google.com/github/jalammar/ecco/blob/main/notebooks/Language_Models_and_Ecco_PyData_Khobar.ipynb">Colab Notebook</a>]


## How-to Guides
- [Interfaces for Explaining Transformer Language Models](https://jalammar.github.io/explaining-transformers/)
- [Finding the Words to Say: Hidden State Visualizations for Language Models](https://jalammar.github.io/hidden-states/)


## API Reference
The [API reference](https://ecco.readthedocs.io/en/main/api/ecco/) and the [architecture](https://ecco.readthedocs.io/en/main/architecture/) page explain Ecco's components and how they work together.

## Gallery & Examples

<div class="container gallery" markdown="1">

<p><strong>Predicted Tokens:</strong> View the model's prediction for the next token (with probability scores). See how the predictions evolved through the model's layers. [<a href="https://github.com/jalammar/ecco/blob/main/notebooks/Ecco_Output_Token_Scores.ipynb">Notebook</a>] [<a href="https://colab.research.google.com/github/jalammar/ecco/blob/main/notebooks/Ecco_Output_Token_Scores.ipynb">Colab</a>]</p>
<img src="docs/img/layer_predictions_ex_london.png" width="400" />
<hr />
<p><strong>Rankings across layers:</strong> After the model picks an output token, Look back at how each layer ranked that token.  [<a href="https://github.com/jalammar/ecco/blob/main/notebooks/Ecco_Evolution_of_Selected_Token.ipynb">Notebook</a>] [<a href="https://colab.research.google.com/github/jalammar/ecco/blob/main/notebooks/Ecco_Evolution_of_Selected_Token.ipynb">Colab</a>]</p>
<img src="docs/img/rankings_ex_eu_1_widethumb.png" width="400"/>
<hr />
<p><strong>Layer Predictions:</strong>Compare the rankings of multiple tokens as candidates for a certain position in the sequence.  [<a href="https://github.com/jalammar/ecco/blob/main/notebooks/Ecco_Comparing_Token_Rankings.ipynb">Notebook</a>] [<a href="https://colab.research.google.com/github/jalammar/ecco/blob/main/notebooks/Ecco_Comparing_Token_Rankings.ipynb">Colab</a>]</p>
<img src="docs/img/rankings_watch_ex_is_are_widethumb.png" width="400" />
<hr />
<p><strong>Primary Attributions:</strong> How much did each input token contribute to producing the output token?   [<a href="https://github.com/jalammar/ecco/blob/main/notebooks/Ecco_Primary_Attributions.ipynb">Notebook</a>] [<a href="https://colab.research.google.com/github/jalammar/ecco/blob/main/notebooks/Ecco_Primary_Attributions.ipynb">Colab</a>]
</p>
<img src="docs/img/saliency_ex_1_thumbwide.png" width="400"/>

<hr />
<p><strong>Detailed Primary Attributions:</strong> See more precise input attributions values using the detailed view. [<a href="https://github.com/jalammar/ecco/blob/main/notebooks/Ecco_Primary_Attributions.ipynb">Notebook</a>] [<a href="https://colab.research.google.com/github/jalammar/ecco/blob/main/notebooks/Ecco_Primary_Attributions.ipynb">Colab</a>]
</p>
<img src="docs/img/saliency_ex_2_thumbwide.png" width="400"/>

<hr />
<p><strong>Neuron Activation Analysis:</strong> Examine underlying patterns in neuron activations using non-negative matrix factorization. [<a href="https://github.com/jalammar/ecco/blob/main/notebooks/Ecco_Neuron_Factors.ipynb">Notebook</a>] [<a href="https://colab.research.google.com/github/jalammar/ecco/blob/main/notebooks/Ecco_Neuron_Factors.ipynb">Colab</a>]</p>
<img src="docs/img/nmf_ex_1_widethumb.png" width="400"/>

</div>

## Getting Help
Having trouble?

- The [Discussion](https://github.com/jalammar/ecco/discussions) board might have some relevant information. If not, you can post your questions there.
- Report bugs at Ecco's [issue tracker](https://github.com/jalammar/ecco/issues)



Bibtex for citations:
```bibtex
@inproceedings{alammar-2021-ecco,
    title = "Ecco: An Open Source Library for the Explainability of Transformer Language Models",
    author = "Alammar, J",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing: System Demonstrations",
    year = "2021",
    publisher = "Association for Computational Linguistics",
}
```
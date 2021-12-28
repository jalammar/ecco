
<img src="https://ar.pegg.io/img/ecco-logo-480.png" width="400" style="background-color: white" />

<br />
<br />

[![PyPI Package latest release](https://img.shields.io/pypi/v/ecco.svg)](https://pypi.org/project/ecco)
[![Supported versions](https://img.shields.io/pypi/pyversions/ecco.svg)](https://pypi.org/project/ecco)


Ecco is a python library for exploring and explaining Natural Language Processing models using interactive visualizations. 


Ecco provides multiple interfaces to aid the explanation and intuition of [Transformer](https//jalammar.github.io/illustrated-transformer/)-based language models. Read: [Interfaces for Explaining Transformer Language Models](https://jalammar.github.io/explaining-transformers/).

Ecco runs inside Jupyter notebooks. It is built on top of [pytorch](https://pytorch.org/) and [transformers](https://github.com/huggingface/transformers).


Ecco is not concerned with training or fine-tuning models. Only exploring and understanding existing pre-trained models. The library is currently an alpha release of a research project. You're welcome to contribute to make it better!


Documentation: [ecco.readthedocs.io](https://ecco.readthedocs.io/)



## Examples:

### What is the sentiment of this film review?
![Sentiment example](https://ar.pegg.io/img/ecco/ecco-sentiment-1.png)

Use a large language model (T5 in this case) to detect text sentiment. In addition to the sentiment, see the tokens the model broke the text into (which can help debug some edge cases).

### Which words in this review lead the model to classify its sentiment as "negative"?
![Feature attribution for sentiment](https://ar.pegg.io/img/ecco/ecco-attrib-ig-1.png)

Feature attribution using Integrated Gradients helps you explore model decisions. In this case, switching "weakness" to "inclination" allows the model to correctly switch the prediction to *positive*.

### Explore the world knowledge of GPT models by posing fill-in-the blank questions.
![Asking GPT2 where heathrow airport is](https://ar.pegg.io/img/ecco/gpt2-heathrow-1.png)

Does GPT2 know where Heathrow Airport is? Yes. It does.

### What other cities/words did the model consider in addition to London?
![Asking GPT2 where heathrow airport is](https://ar.pegg.io/img/ecco/gpt-candidate-logits.png)

Visuals the candidate output tokens and their probability scores.

### Which input words lead it to think of London?
![Asking GPT2 where heathrow airport is](https://ar.pegg.io/img/ecco/heathrow-attribution.png)


### At which layers did the model gather confidence that London is the right answer?
![Asking GPT2 where heathrow airport is](https://ar.pegg.io/img/ecco/token-evolution.png)

The model chose London by making the highest probability token (ranking it #1) after the last layer in the model. How much did each layer contribute to increasing the ranking of *London*? This is a [logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) visualizations that helps explore the activity of different model layers.

### What are the patterns in BERT neuron activation when it processes a piece of text? 

![Asking GPT2 where heathrow airport is](https://ar.pegg.io/img/ecco/neuron-bert.png)

A group of neurons in BERT tend to fire in response to commas and other punctuation. Other groups of neurons tend to fire in response to pronouns. Use this visualization to factorize neuron activity in individual FFNN layers or in the entire model.


Read the paper: 
>[Ecco: An Open Source Library for the Explainability of Transformer Language Models](https://aclanthology.org/2021.acl-demo.30/)
> Association for Computational Linguistics (ACL) System Demonstrations, 2021


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
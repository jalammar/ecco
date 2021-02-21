One of the ways in which Ecco tries to make Transformer language models more transparent is by making it easier to [examine the neuron activations](https://jalammar.github.io/explaining-transformers/) in the feed-forward neural network sublayer of [Transformer blocks](https://jalammar.github.io/illustrated-transformer/). 
Large language models can have up to billions of neurons. Direct examination of these neurons is not always insightful because their firing is sparse, there's a lot of redundancy, and their number makes it hard to extract a signal.

[Matrix decomposition](https://scikit-learn.org/stable/modules/decomposition.html) methods can give us a glimpse into the underlying patterns in neuron firing. From these methods, Ecco currently provides easy access to Non-negative Matrix Factorization (NMF).

![NMF Example](img/nmf_ex_1.png)
## NMF

::: ecco.output.NMF
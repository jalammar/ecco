Ecco works as a wrapper around HuggingFace models. You can load a model from the Model Hub, as well load a local model. model-config.yaml contains configurations for a number of models that Eecco now supports, these include GPT2, BERT, RoBERTa, Albert, Electra, GPTNeo, and T5.

## Loading a local model
To load a local model, prepare its configuration file and pass that along with the model's path to `ecco.from_pretrained()`.

 ```python
    import ecco

    model_config = {
        'embedding': "transformer.wte.weight",
        'type': 'causal',
        'activations': ['mlp\.c_proj'],
        'token_prefix': ' ',
        'partial_token_prefix': ''
    }
    lm = ecco.from_pretrained('gpt2', model_config=model_config)
 ```

Make sure that both the model and its tokenizer are saved in the same path you're passing (the model will typically be a `pytorch_model.bin` file while the tokenizer will have files like `tokenizer.json`, `tokenizer_config.json`, and `vocab.json`).

### Configuration parameters
Ecco uses configuration parameters for its functionality. If trying to load a local model that builds on an existing model (a BERT finetune, for example), you can use the same configuration as BERT. If the model has custom layer names or a custom tokenizer, you'll need to pass the appropriate config parameters.

**'embedding'**: The name of the embeddings layer of the model. This is used to calculate gradient saliency. To get the name of the embedding layer of your model, Running `print(model)` shows the layers of the model. 

**'type'**: the options are 'causal' for GPT-like decoder-based models. 'mlm' for BERT-like encoder-based models. 'enc-dec' for encoder-decoder models like T5.

**'activations'**: The name of the Feed-forward neural network layer inside transformer blocks. This is used for capturing neuron activations.

**token-prefix**: Here we specify the characters that the tokenizer places in the beginning of tokens, if any. GPT2's tokenizer, for example, has a space ' '. BERT does not have any characers.

**partial_token_prefix**: The characters that the tokenizer places in the beginning of partial tokens. BERT, for example, places '##' at the beginning of a partial token.
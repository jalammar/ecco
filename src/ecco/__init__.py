"""
This is main entry point to Ecco. `from_pretrained()` is used to initialize an [LM][ecco.lm.LM]
object which then we use as a language model like GPT2 (or masked language model like BERT).

Usage:

```
    import ecco

    lm = ecco.from_pretrained('distilgpt2')
```
"""


__version__ = '0.1.2'
from ecco.lm import LM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForSeq2SeqLM
from typing import Any, Dict, Optional, List
from ecco.util import load_config, pack_tokenizer_config


def from_pretrained(hf_model_id: str,
                    model_config: Optional[Dict[str, Any]] = None,
                    activations: Optional[bool] = False,
                    attention: Optional[bool] = False,
                    hidden_states: Optional[bool] = True,
                    activations_layer_nums: Optional[List[int]] = None,
                    verbose: Optional[bool] = True,
                    gpu: Optional[bool] = True
                    ):
    """
    Constructs a [LM][ecco.lm.LM] object based on a string identifier from HuggingFace Transformers. This is
    the main entry point to Ecco.

    Usage:

    ```python
    import ecco
    lm = ecco.from_pretrained('gpt2')
    ```

    You can also use a custom model and specify its configurations:
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

    Args:
        hf_model_id (str): Name of the model identifying it in the HuggingFace model hub. e.g. 'distilgpt2', 'bert-base-uncased'.
        model_config (Optional[Dict[str, Any]]): Custom model configuration. If the value is None the config file will be
                                                 searched in the model-config.yaml. Defaults to None.
        activations (Optional[bool]): If True, collect activations when this model runs inference. Option saved in LM. Defaults to False.
        attention (Optional[bool]): If True, collect attention. Option passed to the model. Defaults to False.
        hidden_states (Optional[bool]): If True, collect hidden states. Needed for layer_predictions and rankings(). Defaults to True.
        activations_layer_nums (Optional[List[int]]): If we are collecting activations, we can specify which layers to track. This is None by
                                                      default and all layer are collected if 'activations' is set to True. Defaults to None.
        verbose (Optional[bool]): If True, model.generate() displays output tokens in HTML as they're generated. Defaults to True.
        gpu (Optional[bool]): Set to False to force using the CPU even if a GPU exists. Defaults to True.
    """

    if model_config:
        config = pack_tokenizer_config(model_config)
    else:
        config = load_config(hf_model_id)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)

    if config['type'] == 'enc-dec':
        model_cls = AutoModelForSeq2SeqLM
    elif config['type'] == 'causal':
        model_cls = AutoModelForCausalLM
    else:
        model_cls = AutoModel

    model = model_cls.from_pretrained(hf_model_id, output_hidden_states=hidden_states, output_attentions=attention)

    lm_kwargs = {
        'model_name': hf_model_id,
        'config': config,
        'collect_activations_flag': activations,
        'collect_activations_layer_nums': activations_layer_nums,
        'verbose': verbose,
        'gpu': gpu}

    lm = LM(model, tokenizer, **lm_kwargs)

    return lm

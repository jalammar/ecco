"""
This is main entry point to Ecco. `from_pretrained()` is used to initialize an [LM][ecco.lm.LM]
object which then we use as a language model like GPT2 (or masked language model like BERT).

Usage:

```
    import ecco

    lm = ecco.from_pretrained('distilgpt2')
```


"""


__version__ = '0.0.14'
from ecco.lm import LM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from typing import Optional, List


def from_pretrained(hf_model_id: str,
                    activations: Optional[bool] = False,
                    attention: Optional[bool] = False,
                    hidden_states: Optional[bool] = True,
                    activations_layer_nums: Optional[List[int]] = None,
                    verbose: Optional[bool] = True,
                    gpu: Optional[bool] = True
                    ):
    """
Constructs a [LM][ecco.lm.LM] object based on a string identifier from HuggingFace Transformers. This is main entry point to Ecco.

Usage:

```python
import ecco
lm = ecco.from_pretrained('gpt2')
```

Args:
    hf_model_id: name of the model identifying it in the HuggingFace model hub. e.g. 'distilgpt2', 'bert-base-uncased'.
    activations: If True, collect activations when this model runs inference. Option saved in LM.
    attention: If True, collect attention. Option passed to the model.
    hidden_states: if True, collect hidden states. Needed for layer_predictions and rankings().
    activations_layer_nums: If we are collecting activations, we can specify which layers to track. This is None by
        default and all layer are collected if 'activations' is set to True.
    verbose: If True, model.generate() displays output tokens in HTML as they're generated.
    gpu: Set to False to force using the CPU even if a GPU exists.
"""
    # TODO: Should specify task/head in a cleaner way. Allow masked LM. T5 generation.
    # Likely use model-config. Have a default. Allow user to specify head?
    if 'gpt2' not in hf_model_id:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        model = AutoModel.from_pretrained(hf_model_id,
                                                     output_hidden_states=hidden_states,
                                                     output_attentions=attention)
    else:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        model = AutoModelForCausalLM.from_pretrained(hf_model_id,
                                                     output_hidden_states=hidden_states,
                                                     output_attentions=attention)

    lm_kwargs = {
        'model_name': hf_model_id,
        'collect_activations_flag': activations,
        'collect_activations_layer_nums': activations_layer_nums,
        'verbose': verbose,
        'gpu': gpu}
    lm = LM(model, tokenizer, **lm_kwargs)
    return lm

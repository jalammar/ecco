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
import pathlib
import yaml

from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForSeq2SeqLM
from ecco.lm import LM


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
    path = pathlib.Path(__file__).parent.resolve() / "model-config.yaml"
    configs = yaml.safe_load(path.open())

    if configs[hf_model_id]['type'] == 'enc-dec':
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(hf_model_id,
                                                      output_hidden_states=hidden_states,
                                                      output_attentions=attention)
    elif configs[hf_model_id]['type'] == 'causal':
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        model = AutoModelForCausalLM.from_pretrained(hf_model_id,
                                          output_hidden_states=hidden_states,
                                          output_attentions=attention)
    else:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        model = AutoModel.from_pretrained(hf_model_id,
                                            output_hidden_states=hidden_states,
                                            output_attentions=attention)
    lm_kwargs = {
        'model_name': hf_model_id,
        'config': configs,
        'collect_activations_flag': activations,
        'collect_activations_layer_nums': activations_layer_nums,
        'verbose': verbose,
        'gpu': gpu}
    lm = LM(model, tokenizer, **lm_kwargs)
    return lm


if __name__ == '__main__':
    # Only kept here so I can start debugging conveniently
    lm = from_pretrained('distilgpt2')
    text = ''' Now I ask you: what can be expected of man since he is a being endowed with strange qualities? Shower upon him every earthly blessing, drown him in a sea of happiness, so that nothing but bubbles of bliss can be seen on the surface; give him economic prosperity, such that he should have nothing else to do but sleep, eat cakes and busy himself with the continuation of his species, and even then out of sheer ingratitude, sheer spite, man would play you some nasty trick. He would even risk his cakes and would deliberately desire the most fatal rubbish, the most uneconomical absurdity, simply to introduce into all this positive good sense his fatal fantastic element. It is just his fantastic dreams, his vulgar folly that he will desire to retain, simply in order to prove to himself--as though that were so necessary-- that men still are men and not the keys of a piano, which the laws of nature threaten to control so completely that soon one will be able to desire nothing but by the calendar. And that is not all: even if man really were nothing but a piano-key, even if this were proved to him by natural science and mathematics, even then he would not become reasonable, but would purposely do something perverse out of simple ingratitude, simply to gain his point. And if he does not find means he will contrive destruction and chaos, will contrive sufferings of all sorts, only to gain his point! He will launch a curse upon the world, and as only man can curse (it is his privilege, the primary distinction between him and other animals), may be by his curse alone he will attain his object--that is, convince himself that he is a man and not a piano-key!
'''
    output = lm.generate(text, generate=2, do_sample=True)
__version__ = '0.0.14'
from ecco.lm import LM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from typing import Optional


def from_pretrained(hf_model_id: str,
                    activations: Optional[bool] = False,
                    attention: Optional[bool] = False,
                    hidden_states: Optional[bool] = True,
                    activations_layer_nums: Optional[bool] = None,
                    verbose: Optional[bool] = True
                    ):

    if 'bert' in hf_model_id:
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
        'verbose': verbose}
    lm = LM(model, tokenizer, **lm_kwargs)
    return lm

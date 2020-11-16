__version__ = '0.0.7'
from ecco.language_model.lm import LM
from transformers import AutoTokenizer, AutoModelForCausalLM

def from_pretrained(hf_model_id, activations=False, attention=False):
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    model = AutoModelForCausalLM.from_pretrained(hf_model_id,
                                                 output_hidden_states=True,
                                                 output_attentions=attention)
    if activations:
        return LM(model, tokenizer, collect_activations_flag=True)
    else:
        return LM(model, tokenizer)

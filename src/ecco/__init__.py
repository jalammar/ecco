__version__ = '0.0.10'
from ecco.lm import LM, MockGPT, MockGPTTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

def from_pretrained(hf_model_id, activations=False, attention=False):
    if hf_model_id == "mockGPT":
        tokenizer = MockGPTTokenizer()
        model = MockGPT()
    else:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        model = AutoModelForCausalLM.from_pretrained(hf_model_id,
                                                     output_hidden_states=True,
                                                     output_attentions=attention)
    if activations:
        lm = LM(model, tokenizer, collect_activations_flag=True)
        return lm
    else:
        lm = LM(model, tokenizer)
        return lm

__version__ = '0.0.10'
from ecco.lm import LM, MockGPT, MockGPTTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

def from_pretrained(hf_model_id,
                    activations=False,
                    attention=False,
                    hidden_states=True,
                    activations_layer_nums=None,
                    ):
    if hf_model_id == "mockGPT":
        tokenizer = MockGPTTokenizer()
        model = MockGPT()
    else:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        model = AutoModelForCausalLM.from_pretrained(hf_model_id,
                                                     output_hidden_states=hidden_states,
                                                     output_attentions=attention)

    lm_kwargs = {
        'collect_activations_flag': activations,
        'collect_activations_layer_nums': activations_layer_nums}
    lm = LM(model, tokenizer, **lm_kwargs)
    return lm

__version__ = '0.0.14'
from ecco.lm import LM, MockGPT
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

def from_pretrained(hf_model_id,
                    activations=False,
                    attention=False,
                    hidden_states=True,
                    activations_layer_nums=None,
                    ):
    if hf_model_id == "mockGPT":
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = MockGPT()
    elif 'bert' in hf_model_id:
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
        'collect_activations_layer_nums': activations_layer_nums}
    lm = LM(model, tokenizer, **lm_kwargs)
    return lm

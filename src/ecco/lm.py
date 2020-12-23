import torch
import transformers
import ecco
from torch.nn import functional as F
import numpy as np
from ecco.output import OutputSeq
import random
from IPython import display as d
import os
import json
from ecco.attribution import *
from typing import Optional, Any
from transformers.modeling_gpt2 import GPT2Model


def sample_output_token(scores, do_sample, temperature, top_k, top_p):
    if do_sample:
        # Temperature (higher temperature => more likely to sample low probability tokens)
        if temperature != 1.0:
            scores = scores / temperature
        # Top-p/top-k filtering
        next_token_logscores = transformers.generation_utils. \
            top_k_top_p_filtering(scores,
                                  top_k=top_k,
                                  top_p=top_p)
        # Sample
        probs = F.softmax(next_token_logscores, dim=-1)
        # print(probs.shape)
        prediction_id = torch.multinomial(probs, num_samples=1)
    else:
        # Greedy decoding
        prediction_id = torch.argmax(scores, dim=-1)
    return prediction_id


def _one_hot(token_ids, vocab_size):
    return torch.zeros(len(token_ids), vocab_size).scatter_(1, token_ids.unsqueeze(1), 1.)


def activations_dict_to_array(activations_dict):
    # print(activations_dict[0].shape)
    activations = []
    for i in range(len(activations_dict)):
        activations.append(activations_dict[i])

    activations = np.squeeze(np.array(activations))
    return np.swapaxes(activations, 1, 2)


class LM(object):
    """
    Wrapper around language model. Provides saliency for generated tokens and collects neuron activations.
    """

    def __init__(self, model, tokenizer,
                 collect_activations_flag=False,
                 collect_gen_activations_flag=False):
        self.model = model
        if torch.cuda.is_available():
            self.model = model.to('cuda')

        self.tokenizer = tokenizer
        self._path = os.path.dirname(ecco.__file__)

        self.device = 'cuda' if torch.cuda.is_available() and self.model.device.type == 'cuda' \
            else 'cpu'

        # Neuron Activation
        self.collect_activations_flag = collect_activations_flag
        self.collect_gen_activations_flag = collect_gen_activations_flag
        self._hooks = {}
        self._reset()
        self._attach_hooks(self.model)

        # If running in Jupyer, outputting setup this in one cell is enough. But for colab
        # we're running it before every d.HTML cell
        # d.display(d.HTML(filename=os.path.join(self._path, "html", "setup.html")))

    def _reset(self):
        self._all_activations_dict = {}
        self._generation_activations_dict = {}
        self.activations = []
        self.all_activations = []
        self.generation_activations = []
        self.neurons_to_inhibit = {}
        self.neurons_to_induce = {}

    def to(self, tensor: torch.Tensor):
        if self.device == 'cuda':
            return tensor.to('cuda')
        return tensor

    def _generate_token(self, input_ids, past, do_sample: bool, temperature: float, top_k: int, top_p: float,
                        attribution_flag: Optional[bool]):
        """
        Run a forward pass through the model and sample a token.
        """
        inputs_embeds, token_ids_tensor_one_hot = self._get_embeddings(input_ids)

        output = self.model(inputs_embeds=inputs_embeds, return_dict=True, use_cache=False)
        predict = output[0]
        past = output[1]  # We're not using past because by presenting all the past tokens at every
        # step, we can get feature importance attribution. Let me know if it can be done with past

        scores = predict[-1, :]

        prediction_id = sample_output_token(scores, do_sample, temperature, top_k, top_p)
        # Print the sampled token
        # print(self.tokenizer.decode([prediction_id]))

        # prediction_id now has the id of the token we want to output
        # To do feature importance, let's get the actual logit associated with
        # this token
        prediction_logit = predict[inputs_embeds.shape[0] - 1][prediction_id]

        if attribution_flag:
            saliency_scores = saliency(prediction_logit, token_ids_tensor_one_hot)
            if 'gradient' not in self.attributions:
                self.attributions['gradient'] = []
            self.attributions['gradient'].append(saliency_scores.cpu().detach().numpy())

            grad_x_input = gradient_x_inputs_attribution(prediction_logit,
                                                         inputs_embeds)
            if 'grad_x_input' not in self.attributions:
                self.attributions['grad_x_input'] = []
            self.attributions['grad_x_input'].append(grad_x_input.cpu().detach().numpy())

        return prediction_id, output, past

    def generate(self, input_str: str, max_length: Optional[int] = 128,
                 temperature: Optional[float] = None,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 get_model_output: Optional[bool] = False,
                 do_sample: Optional[bool] = None,
                 attribution: Optional[bool] = True,
                 generate: Optional[int] = None):

        top_k = top_k if top_k is not None else self.model.config.top_k
        top_p = top_p if top_p is not None else self.model.config.top_p
        temperature = temperature if temperature is not None else self.model.config.temperature
        do_sample = do_sample if do_sample is not None else self.model.config.task_specific_params['text-generation'][
            'do_sample']

        input_ids = self.tokenizer(input_str, return_tensors="pt")['input_ids'][0]
        n_input_tokens = len(input_ids)

        if generate is not None:
            max_length = n_input_tokens + generate

        past = None
        self.attributions = {}
        outputs = []

        cur_len = len(input_ids)

        if cur_len >= max_length:
            raise (ValueError,
                   "max_length set to {} while input token has more tokens ({}). Consider increasing max_length" \
                   .format(max_length, cur_len))

        # Print output
        viz_id = self.display_input_sequence(input_ids)

        while cur_len < max_length:
            output_token_id, output, past = self._generate_token(input_ids,
                                                                 past,
                                                                 # Note, this is not currently used
                                                                 temperature=temperature,
                                                                 top_k=top_k, top_p=top_p,
                                                                 do_sample=do_sample,
                                                                 attribution_flag=attribution)

            if (get_model_output):
                outputs.append(output)
            input_ids = torch.cat([input_ids, torch.tensor([output_token_id])])

            self.display_token(viz_id,
                               output_token_id.cpu().numpy(),
                               cur_len)
            cur_len = cur_len + 1

            if output_token_id == self.model.config.eos_token_id:
                break

        # Turn activations from dict to a proper array
        activations_dict = self._all_activations_dict or self._generation_activations_dict

        if activations_dict != {}:
            self.activations = activations_dict_to_array(activations_dict)

        hidden_states = output[2]
        tokens = []
        for i in input_ids:
            token = self.tokenizer.decode([i])
            tokens.append(token)

        attributions = self.attributions
        attn = None
        if len(output) == 4:
            attn = output[-1]
        return OutputSeq(**{'tokenizer': self.tokenizer,
                            'token_ids': input_ids,
                            'n_input_tokens': n_input_tokens,
                            'output_text': self.tokenizer.decode(input_ids),
                            'tokens': tokens,
                            'hidden_states': hidden_states,
                            'attention': attn,
                            'model_outputs': outputs,
                            'attribution': attributions,
                            'activations': self.activations,
                            'lm_head': self.model.lm_head,
                            'device': self.device})

    def _get_embeddings(self, input_ids):
        """
        Takes the token ids of a sequence, returnsa matrix of their embeddings.
        """
        embedding_matrix = self.model.transformer.wte.weight

        vocab_size = embedding_matrix.shape[0]
        one_hot_tensor = self.to(_one_hot(input_ids, vocab_size))

        token_ids_tensor_one_hot = one_hot_tensor.clone().requires_grad_(True)
        # token_ids_tensor_one_hot.requires_grad_(True)

        inputs_embeds = torch.matmul(token_ids_tensor_one_hot, embedding_matrix)
        return inputs_embeds, token_ids_tensor_one_hot

    def _attach_hooks(self, model):
        for name, module in model.named_modules():
            # Add hooks to capture activations in every FFNN
            if "mlp.c_proj" in name:
                if self.collect_activations_flag:
                    self._hooks[name] = module.register_forward_hook(
                        lambda self_, input_, output,
                               name=name: self._get_activations_hook(name, input_))

                if self.collect_gen_activations_flag:
                    self._hooks[name] = module.register_forward_hook(
                        lambda self_, input_, output,
                               name=name: self._get_generation_activations_hook(name, input_))

                # Register neuron inhibition hook
                self._hooks[name + '_inhibit'] = module.register_forward_pre_hook(
                    lambda self_, input_, name=name: \
                        self._inhibit_neurons_hook(name, input_)
                )

    def _get_activations_hook(self, name: str, input_):
        """
        Collects the activation for all tokens (input and output)
        """
        # print(input_.shape, output.shape)
        # in distilGPT and GPT2, the layer name is 'transformer.h.0.mlp.c_fc'
        # Extract the number of the layer from the name
        layer_number = int(name.split('.')[2])

        if layer_number not in self._all_activations_dict:
            self._all_activations_dict[layer_number] = [0]

        # Overwrite the previous step activations. This collects all activations in the last step
        # Assuming all input tokens are presented as input, no "past"
        # The inputs to c_proj already pass through the gelu activation function
        self._all_activations_dict[layer_number][0] = input_[0][0].detach().cpu().numpy()

    def _get_generation_activations_hook(self, name: str, input_):
        """
        Collects the activation for the token being generated
        """
        # print(input_.shape, output.shape)
        # in distilGPT and GPT2, the layer name is 'transformer.h.0.mlp.c_fc'
        # Extract the number of the layer from the name
        layer_number = int(name.split('.')[2])

        if layer_number not in self._generation_activations_dict:
            self._generation_activations_dict[layer_number] = []

        # Accumulate in dict
        # The inputs to c_proj already pass through the gelu activation function
        self._generation_activations_dict[layer_number].append(input_[0][0][-1].detach().cpu().numpy())

    def _inhibit_neurons_hook(self, name: str, input_tensor):
        """
        After being attached as a pre-forward hook, it sets to zero the activation value
        of the neurons indicated in self.neurons_to_inhibit
        """

        layer_number = int(name.split('.')[2])
        if layer_number in self.neurons_to_inhibit.keys():
            # print('layer_number', layer_number, input_tensor[0].shape)

            for n in self.neurons_to_inhibit[layer_number]:
                # print('inhibiting', layer_number, n)
                input_tensor[0][0][-1][n] = 0  # tuple, batch, position

        if layer_number in self.neurons_to_induce.keys():
            # print('layer_number', layer_number, input_tensor[0].shape)

            for n in self.neurons_to_induce[layer_number]:
                # print('inhibiting', layer_number, n)
                input_tensor[0][0][-1][n] = input_tensor[0][0][-1][n] * 10  # tuple, batch, position

        return input_tensor

    def display_input_sequence(self, input_ids):

        tokens = []
        for idx, token_id in enumerate(input_ids):
            type = "input"
            tokens.append({'token': self.tokenizer.decode([token_id]),
                           'position': idx,
                           'token_id': int(token_id),
                           'type': type})
        data = {'tokens': tokens}

        d.display(d.HTML(filename=os.path.join(self._path, "html", "setup.html")))
        d.display(d.HTML(filename=os.path.join(self._path, "html", "basic.html")))
        viz_id = f'viz_{round(random.random() * 1000000)}'
#         html = f"""
# <div id='{viz_id}_output'></div>
# <script>
# """

        js = f"""

         requirejs( ['basic', 'ecco'], function(basic, ecco){{
            basic.init('{viz_id}')

            window.ecco['{viz_id}'] = ecco.renderOutputSequence('{viz_id}', {data})
         }}, function (err) {{
            console.log(err);
        }})
"""
        # print(js)
        # d.display(d.HTML(html))
        d.display(d.Javascript(js))
        return viz_id

    def display_token(self, viz_id, token_id, position):
        token = {
            'token': self.tokenizer.decode([token_id]),
            'token_id': int(token_id),
            'position': position,
            'type': 'output'
        }
        js = f"""
        // We don't really need these require scripts. But this is to avert
        //this code from running before display_input_sequence which DOES require external files
        requirejs(['basic', 'ecco'], function(basic, ecco){{
                console.log('addToken viz_id', '{viz_id}');
                window.ecco['{viz_id}'].addToken({json.dumps(token)})
                window.ecco['{viz_id}'].redraw()
        }})
        """
        # print(js)
        d.display(d.Javascript(js))

    def predict_token(self, inputs, topk=50, temperature=1.0):

        output = self.model(**inputs)
        scores = output[0][0][-1] / temperature
        s = scores.detach().numpy()
        sorted_predictions = s.argsort()[::-1]
        sm = F.softmax(scores, dim=-1).detach().numpy()

        tokens = [self.tokenizer.decode([t]) for t in sorted_predictions[:topk]]
        probs = sm[sorted_predictions[:topk]]

        prediction_data = []
        for idx, (token, prob) in enumerate(zip(tokens, probs)):
            # print(idx, token, prob)
            prediction_data.append({'token': token,
                                    'prob': str(prob),
                                    'ranking': idx + 1,
                                    'token_id': str(sorted_predictions[idx])
                                    })

        params = prediction_data

        viz_id = 'viz_{}'.format(round(random.random() * 1000000))

        d.display(d.HTML(filename=os.path.join(self._path, "html", "predict_token.html")))
        js = """
        requirejs(['predict_token'], function(predict_token){{
        if (window.predict === undefined)
            window.predict = {{}}
        window.predict["{}"] = new predict_token.predictToken("{}", {})
        }}
        )
        """.format(viz_id, viz_id, json.dumps(params))
        d.display(d.Javascript(js))


class MockGPT(GPT2Model):
    def __init__(self):
        print('Mock tokenizer init')
        config = transformers.GPT2Config.from_pretrained("gpt2")
        super().__init__(config)
        self.transformer ={'wte':{'weight':  torch.Tensor([])}}

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __call__(self, **kwargs):
        print('calling model', kwargs)
        return OutputSeq(**{'tokenizer': MockGPTTokenizer(),
                            'token_ids': [352, 11, 352, 11, 362],
                            'n_input_tokens': 4,
                            'output_text': ' 1, 1, 2',
                            'tokens': [' 1', ',', ' 1', ',', ' 2'],
                            'hidden_states': [torch.rand(4, 768) for i in range(7)],
                            'attention': None,
                            'model_outputs': None,
                            'attribution': {'gradient': [
                                np.array([0.41861308, 0.13054065, 0.23851791, 0.21232839], dtype=np.float32)],
                                'grad_x_input': [
                                    np.array([0.31678662, 0.18056837, 0.37555906, 0.12708597],
                                             dtype=np.float32)]},
                            'activations': [],
                            'lm_head': torch.nn.Linear(768, 50257, bias=False),
                            'device': 'cpu'})

    # def generate(self, input_str, generate=1, **kwargs):


class MockGPTTokenizer(transformers.PreTrainedTokenizer):
    def __init__(self):
        super().__init__()

        print('Mock model init')

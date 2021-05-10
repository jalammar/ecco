import json
import os
import random
import re
from operator import attrgetter
from typing import Optional, List, Tuple

import torch
import transformers
import yaml
from IPython import display as d
from torch.nn import functional as F
from transformers.modeling_outputs import Seq2SeqLMOutput

import ecco
from ecco.attribution import *
from ecco.output import OutputSeq
import ecco.attribution_enc_dec as attrib_ed


class LM(object):
    """
    Ecco's central class. A wrapper around language models. We use it to run the language models
    and collect important data like input saliency and neuron activations.

    A LM object is typically not created directly by users,
    it is returned by `ecco.from_pretrained()`.

    Usage:

    ```python
    import ecco

    lm = ecco.from_pretrained('distilgpt2')
    output = lm.generate("Hello computer")
    ```
    """

    def __init__(self,
                 model: transformers.PreTrainedModel,
                 tokenizer: transformers.PreTrainedTokenizerFast,
                 model_name: str,
                 collect_activations_flag: Optional[bool] = False,
                 collect_activations_layer_nums: Optional[List[int]] = None,  # None --> collect for all layers
                 verbose: Optional[bool] = True,
                 gpu: Optional[bool] = True
                 ):
        """
        Creates an LM object given a model and tokenizer.

        Args:
            model: HuggingFace Transformers Pytorch language model.
            tokenizer: The tokenizer associated with the model
            model_name: The name of the model. Used to retrieve required settings (like what the embedding layer is called)
            collect_activations_flag: True if we want to collect activations
            collect_activations_layer_nums: If collecting activations, we can use this parameter to indicate which layers
                to track. By default this would be None and we'd collect activations for all layers.
            verbose: If True, model.generate() displays output tokens in HTML as they're generated.
            gpu: Set to False to force using the CPU even if a GPU exists.
        """
        self.model_name = model_name
        self.model = model
        if torch.cuda.is_available() and gpu:
            self.model = model.to('cuda')

        self.device = 'cuda' if torch.cuda.is_available() \
                                and self.model.device.type == 'cuda' \
            else 'cpu'

        self.tokenizer = tokenizer
        self.verbose = verbose
        self._path = os.path.dirname(ecco.__file__)

        # Neuron Activation
        self.collect_activations_flag = collect_activations_flag
        self.collect_activations_layer_nums = collect_activations_layer_nums

        # For each model, this indicates the layer whose activations
        # we will collect
        configs = yaml.safe_load(open(os.path.join(self._path, "model-config.yaml")))

        try:
            self.model_config = configs[self.model_name]
            self.model_embeddings = self.model_config['embedding']
            embeddings_layer_name = self.model_config['embedding']
            embed_retriever = attrgetter(embeddings_layer_name)
            self.model_embeddings = embed_retriever(self.model)
            self.collect_activations_layer_name_sig = self.model_config['activations'][0]
        except KeyError:
            raise ValueError(
                f"The model '{self.model_name}' is not defined in Ecco's 'model-config.yaml' file and"
                f" so is not explicitly supported yet. Supported models are:",
                list(configs.keys())) from KeyError()

        self._hooks = {}
        self._reset()
        self._attach_hooks(self.model)

        # If running in Jupyer, outputting setup this in one cell is enough. But for colab
        # we're running it before every d.HTML cell
        # d.display(d.HTML(filename=os.path.join(self._path, "html", "setup.html")))

    def _reset(self):
        self._all_activations_dict = {}
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
        predict = output.logits

        scores = predict[-1:, :]

        prediction_id = sample_output_token(scores, do_sample, temperature, top_k, top_p)

        # prediction_id now has the id of the token we want to output
        # To do feature importance, let's get the actual logit associated with
        # this token
        prediction_logit = predict[inputs_embeds.shape[0] - 1][prediction_id]

        if attribution_flag:
            saliency_results = compute_saliency_scores(prediction_logit, token_ids_tensor_one_hot, inputs_embeds)

            if 'gradient' not in self.attributions:
                self.attributions['gradient'] = []
            self.attributions['gradient'].append(saliency_results['gradient'].cpu().detach().numpy())

            if 'grad_x_input' not in self.attributions:
                self.attributions['grad_x_input'] = []
            self.attributions['grad_x_input'].append(saliency_results['grad_x_input'].cpu().detach().numpy())

        output['logits'] = None  # free tensor memory we won't use again

        # detach(): don't need grads here
        # cpu(): not used by GPU during generation; may lead to GPU OOM if left on GPU during long generations
        if getattr(output, "hidden_states", None) is not None:
            hs_list = []
            for idx, layer_hs in enumerate(output.hidden_states):
                # in Hugging Face Transformers v4, there's an extra index for batch
                if len(layer_hs.shape) == 3:  # If there's a batch dimension, pick the first oen
                    hs = layer_hs.cpu().detach()[0].unsqueeze(0)  # Adding a dimension to concat to later
                # Earlier versions are only 2 dimensional
                # But also, in v4, for GPT2, all except the last one would have 3 dims, the last layer
                # would only have two dims
                else:
                    hs = layer_hs.cpu().detach().unsqueeze(0)

                hs_list.append(hs)

            output.hidden_states = torch.cat(hs_list, dim=0)

        return prediction_id, output

    def generate(self, input_str: str,
                 max_length: Optional[int] = 8,
                 temperature: Optional[float] = None,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 get_model_output: Optional[bool] = False,
                 do_sample: Optional[bool] = None,
                 attribution: Optional[bool] = True,
                 generate: Optional[int] = None):
        """
        Generate tokens in response to an input prompt.
        Works with Language models like GPT2, not masked language models like BERT.
        Args:
            input_str: Input prompt.
            generate: Number of tokens to generate.
            max_length: max length of sequence (input + output tokens)
            temperature: Adjust the probability distibution of output candidate tokens.
            top_k: Specify top-k tokens to consider in decoding. Only used when do_sample is True.
            top_p: Specify top-p to consider in decoding. Only used when do_sample is True.
            get_model_output:  Flag to retrieve the final output object returned by the underlying language model.
            do_sample: Decoding parameter. If set to False, the model always always
                chooses the highest scoring candidate output
                token. This may lead to repetitive text. If set to True, the model considers
                consults top_k and/or top_p to generate more itneresting output.
            attribution: If True, the object will calculate input saliency/attribution.
        """
        top_k = top_k if top_k is not None else self.model.config.top_k
        top_p = top_p if top_p is not None else self.model.config.top_p
        temperature = temperature if temperature is not None else self.model.config.temperature
        do_sample = do_sample if do_sample is not None else self.model.config.task_specific_params['text-generation'][
            'do_sample']

        # We needs this as a batch in order to collect activations.
        input_ids = self.tokenizer(input_str, return_tensors="pt")['input_ids'][0]
        n_input_tokens = len(input_ids)
        cur_len = n_input_tokens

        if generate is not None:
            max_length = n_input_tokens + generate

        past = None
        self.attributions = {}
        outputs = []

        if cur_len >= max_length:
            raise ValueError(
                "max_length set to {} while input token has more tokens ({}). Consider increasing max_length" \
                    .format(max_length, cur_len))

        # Print output
        if self.verbose:
            viz_id = self.display_input_sequence(input_ids)

        while cur_len < max_length:
            output_token_id, output = self._generate_token(input_ids,
                                                           past,
                                                           # Note, this is not currently used
                                                           temperature=temperature,
                                                           top_k=top_k, top_p=top_p,
                                                           do_sample=do_sample,
                                                           attribution_flag=attribution)

            if get_model_output:
                outputs.append(output)
            input_ids = torch.cat([input_ids, torch.tensor([output_token_id])])

            if self.verbose:
                self.display_token(viz_id,
                                   output_token_id.cpu().numpy(),
                                   cur_len)
            cur_len = cur_len + 1

            if output_token_id == self.model.config.eos_token_id:
                break

        # Turn activations from dict to a proper array
        activations_dict = self._all_activations_dict

        if activations_dict != {}:
            self.activations = activations_dict_to_array(activations_dict)

        hidden_states = getattr(output, "hidden_states", None)
        tokens = []
        for i in input_ids:
            token = self.tokenizer.decode([i])
            tokens.append(token)

        attributions = self.attributions
        attn = getattr(output, "attentions", None)
        return OutputSeq(**{'tokenizer': self.tokenizer,
                            'token_ids': input_ids.unsqueeze(0),  # Add a batch dimension
                            'n_input_tokens': n_input_tokens,
                            'output_text': self.tokenizer.decode(input_ids),
                            'tokens': [tokens],  # Add a batch dimension
                            'hidden_states': hidden_states,
                            'attention': attn,
                            'model_outputs': outputs,
                            'attribution': attributions,
                            'activations': self.activations,
                            'collect_activations_layer_nums': self.collect_activations_layer_nums,
                            'lm_head': self.model.lm_head,
                            'device': self.device})

    def __call__(self,
                 # input_str: Optional[str] = '',
                 input_tokens: torch.Tensor,
                 # attribution: Optional[bool] = True,
                 ):
        """
        Run a forward pass through the model. For when we don't care about output tokens.
        Currently only support activations collection. No attribution/saliency.

        Usage:

        ```python
        inputs = lm.tokenizer("Hello computer", return_tensors="pt")
        output = lm(inputs)
        ```

        Args:
            input_tokens: tuple returned by tokenizer( TEXT, return_tensors="pt").
                contains key 'input_ids', its value tensor with input token ids.
                Shape is (batch_size, sequence_length).
                Also a key for masked tokens
            attribution: Flag indicating whether to calculate attribution/saliency
        """

        if not hasattr(input_tokens, 'input_ids'):
            raise ValueError("Parameter 'input_tokens' needs to have the attribute 'input_ids'."
                             "Verify it was produced by the appropriate tokenizer with the "
                             "parameter return_tensors=\"pt\".")

        # Move inputs to GPU if the model is on GPU
        if self.model.device.type == "cuda" and input_tokens['input_ids'].device.type == "cpu":
            input_tokens = self.to(input_tokens)

        # Remove downstream. For now setting to batch length
        n_input_tokens = len(input_tokens['input_ids'][0])
        # self.attributions = {}

        # model
        if 'bert' in self.model_name:
            output = self.model(**input_tokens, return_dict=True)
            lm_head = None
        else:
            output = self.model(**input_tokens, return_dict=True, use_cache=False)
            predict = output.logits
            scores = predict[-1:, :]
            lm_head = self.model.lm_head

        # Turn activations from dict to a proper array
        activations_dict = self._all_activations_dict
        if activations_dict != {}:
            self.activations = activations_dict_to_array(activations_dict)

        hidden_states = getattr(output, "hidden_states", None)
        tokens = []
        for i in input_tokens['input_ids']:
            token = self.tokenizer.convert_ids_to_tokens(i)
            tokens.append(token)

        attn = getattr(output, "attentions", None)
        return OutputSeq(**{'tokenizer': self.tokenizer,
                            'token_ids': input_tokens['input_ids'],
                            'n_input_tokens': n_input_tokens,
                            # 'output_text': self.tokenizer.decode(input_ids),
                            'tokens': tokens,
                            'hidden_states': hidden_states,
                            'attention': attn,
                            # 'model_outputs': outputs,
                            # 'attribution': attributions,
                            'activations': self.activations,
                            'collect_activations_layer_nums': self.collect_activations_layer_nums,
                            'lm_head': lm_head,
                            'device': self.device})

    def _get_embeddings(self, input_ids):
        """
        Takes the token ids of a sequence, returns a matrix of their embeddings.
        """
        # embedding_matrix = self.model.transformer.wte.weight
        embedding_matrix = self.model_embeddings

        vocab_size = embedding_matrix.shape[0]
        one_hot_tensor = self.to(_one_hot(input_ids, vocab_size))

        token_ids_tensor_one_hot = one_hot_tensor.clone().requires_grad_(True)
        # token_ids_tensor_one_hot.requires_grad_(True)

        inputs_embeds = torch.matmul(token_ids_tensor_one_hot, embedding_matrix)
        return inputs_embeds, token_ids_tensor_one_hot

    def _attach_hooks(self, model):
        for name, module in model.named_modules():
            # Add hooks to capture activations in every FFNN

            if re.search(self.collect_activations_layer_name_sig, name):
                # print("mlp.c_proj", self.collect_activations_flag , name)
                if self.collect_activations_flag:
                    self._hooks[name] = module.register_forward_hook(
                        lambda self_, input_, output,
                               name=name: self._get_activations_hook(name, input_))

                # Register neuron inhibition hook
                self._hooks[name + '_inhibit'] = module.register_forward_pre_hook(
                    lambda self_, input_, name=name: \
                        self._inhibit_neurons_hook(name, input_)
                )

    def _get_activations_hook(self, name: str, input_):
        """
        Collects the activation for all tokens (input and output).
        The default activations collection method.

        Args:
            input_: activation tuple to capture. A tuple containing one tensor of
            dimensions (batch_size, sequence_length, neurons)
        """
        # print('_get_activations_hook', name)
        # pprint(input_)
        # print(type(input_), len(input_), type(input_[0]), input_[0].shape, len(input_[0]), input_[0][0].shape)
        # in distilGPT and GPT2, the layer name is 'transformer.h.0.mlp.c_fc'
        # Extract the number of the layer from the name
        # TODO: it will not always be 2 for other models. Move to model-config
        # layer_number = int(name.split('.')[2])
        # Get the layer number. This will be an int with periods before aand after it.
        # (?<=\.) means look for a period before the int
        # \d+ means look for one or multiple digits
        # (?=\.) means look for a period after the int
        layer_number = re.search("(?<=\.)\d+(?=\.)", name).group(0)
        # print("layer number: ", layer_number)

        collecting_this_layer = (self.collect_activations_layer_nums is None) or (
                layer_number in self.collect_activations_layer_nums)

        if collecting_this_layer:
            # Initialize the layer's key the first time we encounter it
            if layer_number not in self._all_activations_dict:
                self._all_activations_dict[layer_number] = [0]

            # For MLM, we only run one inference step. We save it.
            # For LM, we could be running multiple inference stesp with generate(). In that case,
            # overwrite the previous step activations. This collects all activations in the last step
            # Assuming all input tokens are presented as input, no "past"
            # The inputs to c_proj already pass through the gelu activation function
            self._all_activations_dict[layer_number] = input_[0].detach().cpu().numpy()

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

        prediction_id = torch.multinomial(probs, num_samples=1)
    else:
        # Greedy decoding
        prediction_id = torch.argmax(scores, dim=-1)
    prediction_id = prediction_id.squeeze()
    return prediction_id


def _one_hot(token_ids, vocab_size):
    return torch.zeros(len(token_ids), vocab_size).scatter_(1, token_ids.unsqueeze(1), 1.)


def activations_dict_to_array(activations_dict):
    """
    Converts the dict used to collect activations into an array of the
    shape (batch, layers, neurons, token position).
    Args:
        activations_dict: python dictionary. Contains a key/value for each layer
        in the model whose activations were collected. Key is the layer id ('0', '1').
        Value is a tensor of shape (batch, position, neurons).
    """

    activations = []
    for i in sorted(activations_dict.keys()):
        activations.append(activations_dict[i])

    activations = np.array(activations)
    # 'activations' now is in the shape (layer, batch, position, neurons)

    activations = np.swapaxes(activations, 2, 3)
    activations = np.swapaxes(activations, 0, 1)
    # print('after swapping: ', activations.shape)
    return activations


class T5LM(LM):
    def _attach_hooks(self, model):
        # TODO: Hooks removed for experimentation sake, put them back in once we need them
        pass

    def _get_embeddings(self, input_ids) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Takes the token ids of a sequence, returns a matrix of their embeddings.
        """
        # embedding_matrix = self.model.transformer.wte.weight
        embedding_matrix = self.model_embeddings
        vocab_size = embedding_matrix.shape[0]
        batch_size, num_tokens = input_ids.shape
        one_hot_tensor = torch.zeros(batch_size, num_tokens, vocab_size).scatter_(-1, input_ids.unsqueeze(-1), 1.)
        one_hot_tensor = self.to(one_hot_tensor)
        token_ids_tensor_one_hot = one_hot_tensor.clone().requires_grad_(True)
        # token_ids_tensor_one_hot.requires_grad_(True)

        inputs_embeds = torch.matmul(token_ids_tensor_one_hot, embedding_matrix)
        return inputs_embeds, token_ids_tensor_one_hot

    def generate(self,
                 input_str: str,
                 max_length: Optional[int] = 8,
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
        pad_token_id = self.model.config.pad_token_id
        eos_token_id = self.model.config.eos_token_id

        # We needs this as a batch in order to collect activations.
        input_ids = self.tokenizer(input_str, return_tensors="pt")['input_ids']  # Shape Batch x Tokens
        n_input_tokens = len(input_ids[0])
        cur_len = n_input_tokens

        if generate is not None:
            max_length = n_input_tokens + generate

        past = None
        self.attributions = {}
        outputs = []

        if cur_len >= max_length:
            raise ValueError("max_length set to {} while input token has more tokens ({}). "
                             "Consider increasing max_length".format(max_length, cur_len))

        # Print output
        if self.verbose:
            viz_id = self.display_input_sequence(input_ids[0])

        attention_mask = self.model._prepare_attention_mask_for_generation(input_ids, pad_token_id, eos_token_id)
        decoder_input_ids = self.model._prepare_decoder_input_ids_for_generation(input_ids, None, None)
        # At this point we have encoder_outputs and the encoding part is finished, we now want to start generating
        # We might want to work with decoder_input_ids

        while cur_len < max_length:
            output_token_id, output = self._generate_token(encoder_input_ids=input_ids,
                                                           encoder_attention_mask=attention_mask,
                                                           decoder_input_ids=decoder_input_ids,
                                                           past=past,  # Note, this is not currently used
                                                           temperature=temperature,
                                                           top_k=top_k,
                                                           top_p=top_p,
                                                           do_sample=do_sample,
                                                           attribution_flag=attribution)

            # TODO: vvv These are broken vvv
            if get_model_output:
                outputs.append(output)
            # TODO: ^^^ These are broken ^^^

            decoder_input_ids = torch.cat([decoder_input_ids, torch.tensor([[output_token_id]])], dim=-1)

            if self.verbose:
                self.display_token(viz_id, output_token_id.cpu().numpy(), cur_len)
            cur_len = cur_len + 1

            if output_token_id == self.model.config.eos_token_id:
                break

        # TODO: vvv These are broken vvv
        # Turn activations from dict to a proper array
        activations_dict = self._all_activations_dict
        if activations_dict != {}:
            self.activations = activations_dict_to_array(activations_dict)
        hidden_states = getattr(output, "hidden_states", None)
        attn = getattr(output, "attentions", None)
        # TODO: ^^^ These are broken ^^^

        full_seq_ids = torch.cat([input_ids, decoder_input_ids], dim=-1)
        tokens = []
        for i in full_seq_ids[0]:
            token = self.tokenizer.decode([i])
            tokens.append(token)

        attributions = self.attributions
        return OutputSeq(**{
            'tokenizer': self.tokenizer,
            'token_ids': full_seq_ids,
            'n_input_tokens': n_input_tokens,
            'output_text': self.tokenizer.decode(full_seq_ids[0]),
            'tokens': [tokens],  # Add a batch dimension
            'attribution': attributions,

            # TODO: vvv These are broken vvv
            'hidden_states': hidden_states,
            'attention': attn,
            'model_outputs': outputs,
            'activations': self.activations,
            'collect_activations_layer_nums': self.collect_activations_layer_nums,
            'lm_head': self.model.lm_head,
            # TODO: ^^^ These are broken ^^^

            'device': self.device
        })

    def _generate_token(self,
                        encoder_input_ids,
                        encoder_attention_mask,
                        decoder_input_ids,
                        past,
                        do_sample: bool,
                        temperature: float,
                        top_k: int,
                        top_p: float,
                        attribution_flag: Optional[bool]):
        encoder_inputs_embeds, encoder_token_ids_tensor_one_hot = self._get_embeddings(encoder_input_ids)
        # B x T x E, B x T x V

        # This is okay as long as encoder and decoder share the embeddings
        decoder_inputs_embeds, decoder_token_ids_tensor_one_hot = self._get_embeddings(decoder_input_ids)
        # B x T x E, B x T x V

        output: Seq2SeqLMOutput = self.model(
            # TODO: This re-forwarding encoder side is expensive, can we optimise or is it needed everytime for fresh
            #       backward ?
            inputs_embeds=encoder_inputs_embeds,
            attention_mask=encoder_attention_mask,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=False,
            return_dict=True,
        )
        """Seq2SeqLMOutput has
        
        loss: Optional[torch.FloatTensor]
        logits: torch.FloatTensor
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]
        decoder_hidden_states: Optional[Tuple[torch.FloatTensor]]
        decoder_attentions: Optional[Tuple[torch.FloatTensor]]
        cross_attentions: Optional[Tuple[torch.FloatTensor]]
        encoder_last_hidden_state: Optional[torch.FloatTensor]
        encoder_hidden_states: Optional[Tuple[torch.FloatTensor]]
        encoder_attentions: Optional[Tuple[torch.FloatTensor]]
        """
        predict = output.logits
        scores = predict[0, -1:, :]
        prediction_id = sample_output_token(scores, do_sample, temperature, top_k, top_p)
        # prediction_id now has the id of the token we want to output
        # To do feature importance, let's get the actual logit associated with
        # this token
        prediction_logit = predict[0][-1][prediction_id]

        if attribution_flag:
            saliency_results = attrib_ed.compute_saliency_scores(
                prediction_logit,
                encoder_token_ids_tensor_one_hot,
                encoder_inputs_embeds,
                decoder_token_ids_tensor_one_hot,
                decoder_inputs_embeds,
            )

            if 'gradient' not in self.attributions:
                self.attributions['gradient'] = []
            self.attributions['gradient'].append(saliency_results['gradient'].cpu().detach().numpy())

            if 'grad_x_input' not in self.attributions:
                self.attributions['grad_x_input'] = []
            self.attributions['grad_x_input'].append(saliency_results['grad_x_input'].cpu().detach().numpy())

        output['logits'] = None  # free tensor memory we won't use again

        # detach(): don't need grads here
        # cpu(): not used by GPU during generation; may lead to GPU OOM if left on GPU during long generations
        # TODO: Re-write this for T5
        # if getattr(output, "hidden_states", None) is not None:
        #     hs_list = []
        #     for idx, layer_hs in enumerate(output.hidden_states):
        #         # in Hugging Face Transformers v4, there's an extra index for batch
        #         if len(layer_hs.shape) == 3:  # If there's a batch dimension, pick the first oen
        #             hs = layer_hs.cpu().detach()[0].unsqueeze(0)  # Adding a dimension to concat to later
        #         # Earlier versions are only 2 dimensional
        #         # But also, in v4, for GPT2, all except the last one would have 3 dims, the last layer
        #         # would only have two dims
        #         else:
        #             hs = layer_hs.cpu().detach().unsqueeze(0)
        #
        #         hs_list.append(hs)
        #
        #     output.hidden_states = torch.cat(hs_list, dim=0)

        return prediction_id, output



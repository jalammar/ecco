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
from ..attribution import *


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


class LM(object):
    """

    """

    def __init__(self, model, tokenizer):
        self.model = model
        if torch.cuda.is_available():
            self.model = model.to('cuda')

        self.tokenizer = tokenizer
        self._path = os.path.dirname(ecco.__file__)

        self.device = 'cuda' if torch.cuda.is_available() and self.model.device.type == 'cuda' \
            else 'cpu'
        # torch.device('cuda' if torch.cuda.is_available() and self.model.device.type =='cuda'
        #                        else 'cpu')

        # If running in Jupyer, outputting setup this in one cell is enough. But for colab
        # we're running it before every d.HTML cell
        d.display(d.HTML(filename=os.path.join(self._path, "html", "setup.html")))

    def to(self, tensor):
        if self.device == 'cuda':
            return tensor.to('cuda')
        return tensor

    def _get_embeddings(self, input_ids):
        """
        Takes the token ids of a sequence, returnsa matrix of their embeddings.
        """
        embedding_matrix = self.model.transformer.wte.weight

        vocab_size = embedding_matrix.shape[0]
        one_hot_tensor = self.to(_one_hot(input_ids, vocab_size))

        token_ids_tensor_one_hot = one_hot_tensor.clone().requires_grad_(True)
        token_ids_tensor_one_hot.requires_grad_(True)

        inputs_embeds = torch.matmul(token_ids_tensor_one_hot, embedding_matrix)
        return inputs_embeds, token_ids_tensor_one_hot

    def _generate_token(self, input_ids, past,
                        do_sample, temperature, top_k, top_p, attribution_flag):
        """
        Run a forward pass through the model and sample a token.
        """
        inputs_embeds, token_ids_tensor_one_hot = self._get_embeddings(input_ids)


        # inputs_embeds = inputs_embeds.to('cuda')
        output = self.model(inputs_embeds=inputs_embeds, return_dict=True)
        predict = output[0]
        past = output[1]  # We're not using past because by presenting all the past tokens at every
        # step, we can get feature importance attribution. Let me know if it can be done with past

        scores = predict[-1, :]
        # print(torch.topk(predict[inputs_embeds.shape[0] - 1], 2))

        prediction_id = sample_output_token(scores, do_sample, temperature, top_k, top_p)
        # Print the sampled token
        # print(self.tokenizer.decode([prediction_id]))

        # prediction_id now has the id of the token we want to output
        # To do feature importance, let's get the actual logit associated with
        # this token
        prediction_logit = predict[inputs_embeds.shape[0] - 1][prediction_id]

        if attribution_flag:
            saliency_scores = saliency(prediction_logit, token_ids_tensor_one_hot)
            if 'saliency' not in self.attributions:
                self.attributions['saliency'] = []
            self.attributions['saliency'].append(saliency_scores.cpu().detach().numpy())

            # saliency_scores_2 = saliency(prediction_logit, token_ids_tensor_one_hot, norm=False)
            #
            # if 'saliency_2' not in self.attributions:
            #     self.attributions['saliency_2'] = []
            # self.attributions['saliency_2'].append(saliency_scores_2.cpu().detach().numpy())
            #
            # saliency_scores_embed = saliency_on_d_embeddings(prediction_logit,
            #                                                  inputs_embeds,
            #                                                  aggregation="L2")
            # if 'saliency_embed' not in self.attributions:
            #     self.attributions['saliency_embed'] = []
            # self.attributions['saliency_embed'].append(saliency_scores_embed.cpu().detach().numpy())
            #
            # saliency_scores_embed_sum = saliency_on_d_embeddings(prediction_logit,
            #                                                      inputs_embeds,
            #                                                      aggregation="sum")
            # if 'saliency_embed_sum' not in self.attributions:
            #     self.attributions['saliency_embed_sum'] = []
            # self.attributions['saliency_embed_sum'].append(saliency_scores_embed_sum.cpu().detach().numpy())

            grad_x_input = gradient_x_inputs_attribution(prediction_logit,
                                                         inputs_embeds)
            if 'grad_x_input' not in self.attributions:
                self.attributions['grad_x_input'] = []
            self.attributions['grad_x_input'].append(grad_x_input.cpu().detach().numpy())

        return prediction_id, output, past

    def generate(self, input_str: str, max_length=128,
                 temperature=None, top_k=None, top_p=None, get_model_output=False,
                 do_sample=None, attribution=True, generate=None):

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
        importances = []
        gradientXinputs = []

        cur_len = len(input_ids)

        assert cur_len < max_length, \
            "max_length set to {} while input token has more tokens ({}). Consider increasing max_length" \
                .format(max_length, cur_len)

        while cur_len < max_length:
            output_token_id, output, past = self._generate_token(input_ids,
                                                                 past,
                                                                 # Note, this is not currently used
                                                                 temperature=temperature,
                                                                 top_k=top_k, top_p=top_p,
                                                                 do_sample=do_sample,
                                                                 attribution_flag=attribution)

            if(get_model_output):
                outputs.append(output)
            input_ids = torch.cat([input_ids, torch.tensor([output_token_id])])
            cur_len = cur_len + 1
            if output_token_id == self.model.config.eos_token_id:
                break


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
                            'attribution': attributions})

    def trace_tokens(self, output, position=0, topk=10, layer=None):
        """

        :param output: LMOutput object generated by LM.generate
        :param position: The index of the output token to trace
        :return: Creates a visualization projecting the hidden state output of each layer onto the model's vocabulary
        """

        watch = self.to(torch.tensor([output.token_ids[output.n_input_tokens]]))
        # There is one lm output per generated token. To get the index
        output_index = position - output.n_input_tokens
        # hidden_states = output.outputs[output_index][-2]  # Ignore the first element (embedding)
        if layer is not None:
            hidden_states = output.hidden_states[layer+1].unsqueeze(0)
        else:
            hidden_states = output.hidden_states[1:]  # Ignore the first element (embedding)

        k = topk
        top_tokens = []
        probs = []
        data = []

        print('Predictions for position {}'.format(position))
        for layer_no, h in enumerate(hidden_states):
            # print(h.shape)
            hidden_state = h[position - 1]
            #     print(hidden_state.shape)
            logits = self.model.lm_head(hidden_state)
            softmax = F.softmax(logits, dim=-1)
            sorted_softmax = self.to(torch.argsort(softmax))
            idx = sorted_softmax.shape[0] - torch.nonzero((sorted_softmax == watch)).flatten()

            layer_top_tokens = [self.tokenizer.decode([t]) for t in sorted_softmax[-k:]][::-1]
            top_tokens.append(layer_top_tokens)
            layer_probs = softmax[sorted_softmax[-k:]].cpu().detach().numpy()[::-1]
            probs.append(layer_probs.tolist())

            # Package in output format
            layer_data = []
            for idx, (token, prob) in enumerate(zip(layer_top_tokens, layer_probs)):
                # print(layer_no, idx, token)
                layer_num = layer if layer is not None else layer_no
                layer_data.append({'token': token,
                                   'prob': str(prob),
                                   'ranking': idx + 1,
                                   'layer': layer_num
                                   })

            data.append(layer_data)
            layer_formatted_probs = ['{:02.2f}%'.format(p * 100) for p in layer_probs]

        params = data

        viz_id = 'viz_{}'.format(round(random.random() * 1000000))
        d.display(d.HTML(filename=os.path.join(self._path, "html", "setup.html")))
        d.display(d.HTML(filename=os.path.join(self._path, "html", "trace_tokens.html")))
        js = """
        requirejs(['trace_tokens'], function(trace_tokens){{
        if (window.trace === undefined) 
            window.trace = {{}}
        window.trace["{}"] = new trace_tokens.TraceTokens("{}", {})
        }}
        )
        """.format(viz_id, viz_id, json.dumps(params))
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

import os
import json
import ecco
from IPython import display as d
from ecco import util, lm_plots
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from sklearn import decomposition
from typing import Optional, List


class OutputSeq:
    def __init__(self,
                 token_ids=None,
                 n_input_tokens=None,
                 tokenizer=None,
                 output_text=None,
                 tokens=None,
                 hidden_states=None,
                 attribution=None,
                 activations=None,
                 activations_type=None,
                 attention=None,
                 model_outputs=None,
                 lm_head=None,
                 device='cpu'):
        self.token_ids = token_ids
        self.tokenizer = tokenizer
        self.n_input_tokens = n_input_tokens
        self.output_text = output_text
        self.tokens = tokens
        self.hidden_states = hidden_states
        self.attribution = attribution
        self.activations = activations
        self.activations_type = activations_type
        self.model_outputs = model_outputs
        self.attention_values = attention
        self.lm_head = lm_head
        self.device = device
        self._path = os.path.dirname(ecco.__file__)

    def __str__(self):
        return "<LMOutput '{}' # of lm outputs: {}>".format(self.output_text, len(self.hidden_states))

    def to(self, tensor: torch.Tensor):
        if self.device == 'cuda':
            return tensor.to('cuda')
        return tensor

    def explorable(self, printJson: Optional[bool] = False):

        tokens = []
        for idx, token in enumerate(self.tokens):
            type = "input" if idx < self.n_input_tokens else 'output'

            tokens.append({'token': token,
                           'token_id': int(self.token_ids[idx]),
                           'type': type
                           })

        data = {
            'tokens': tokens
        }

        d.display(d.HTML(filename=os.path.join(self._path, "html", "setup.html")))

        d.display(d.HTML(filename=os.path.join(self._path, "html", "basic.html")))
        viz_id = 'viz_{}'.format(round(random.random() * 1000000))
        js = """
         requirejs(['basic', 'ecco'], function(basic, ecco){{
            const viz_id = basic.init()

            ecco.renderOutputSequence(viz_id, {})
         }}, function (err) {{
            console.log(err);
        }})""".format(data)
        d.display(d.Javascript(js))

        if printJson:
            print(data)

    def __call__(self, position=None, **kwargs):

        if position is not None:
            self.position(position, **kwargs)

        else:
            self.saliency(**kwargs)

    def position(self, position, attr_method='grad_x_input'):

        if (position < self.n_input_tokens) or (position > len(self.tokens) - 1):
            raise ValueError("'position' should indicate a position of a generated token. "
                             "Accepted values for this sequence are between {} and {}."
                             .format(self.n_input_tokens, len(self.tokens) - 1))

        importance_id = position - self.n_input_tokens
        tokens = []
        attribution = self.attribution[attr_method]
        for idx, token in enumerate(self.tokens):
            type = "input" if idx < self.n_input_tokens else 'output'
            if idx < len(attribution[importance_id]):
                imp = attribution[importance_id][idx]
            else:
                imp = -1

            tokens.append({'token': token,
                           'token_id': int(self.token_ids[idx]),
                           'type': type,
                           'value': str(imp)  # because json complains of floats
                           })

        data = {
            'tokens': tokens
        }

        d.display(d.HTML(filename=os.path.join(self._path, "html", "setup.html")))
        d.display(d.HTML(filename=os.path.join(self._path, "html", "basic.html")))
        viz_id = 'viz_{}'.format(round(random.random() * 1000000))
        js = """
         requirejs(['basic', 'ecco'], function(basic, ecco){{
            const viz_id = basic.init()

            ecco.renderSeqHighlightPosition(viz_id, {}, {})
         }}, function (err) {{
            console.log(err);
        }})""".format(position, data)
        d.display(d.Javascript(js))

    def sal(self, **kwargs):
        """Alias for saliency()"""
        self.saliency(**kwargs)

    def saliency(self, attr_method: Optional[str] = 'grad_x_input', **kwargs):
        """
        Explorable showing saliency of each token generation step.
        Hovering-over or tapping an output token imposes a saliency map on other tokens
        showing their importance as features to that prediction.
        """
        position = self.n_input_tokens

        importance_id = position - self.n_input_tokens
        tokens = []
        attribution = self.attribution[attr_method]
        for idx, token in enumerate(self.tokens):
            type = "input" if idx < self.n_input_tokens else 'output'
            if idx < len(attribution[importance_id]):
                imp = attribution[importance_id][idx]
            else:
                imp = 0

            tokens.append({'token': token,
                           'token_id': int(self.token_ids[idx]),
                           'type': type,
                           'value': str(imp),  # because json complains of floats
                           'position': idx
                           })

        data = {
            'tokens': tokens,
            'attributions': [att.tolist() for att in attribution]
        }

        d.display(d.HTML(filename=os.path.join(self._path, "html", "setup.html")))
        d.display(d.HTML(filename=os.path.join(self._path, "html", "basic.html")))
        viz_id = 'viz_{}'.format(round(random.random() * 1000000))
        js = """
         requirejs(['basic', 'ecco'], function(basic, ecco){{
            const viz_id = basic.init()
            ecco.interactiveTokens(viz_id, {})
         }}, function (err) {{
            console.log(err);
        }})""".format(data)
        d.display(d.Javascript(js))

        if 'printJson' in kwargs and kwargs['printJson']:
            print(data)
            return data

    def _repr_html_(self, **kwargs):
        # if util.type_of_script() == "jupyter":
        self.explorable(**kwargs)
        return '<OutputSeq>'
        # else:
        #     return "<OutputSeq Generated tokens: {}. \nFull sentence:'{}' \n# of lm outputus: {}\nTokens:\n{}>" \
        #         .format(self.tokens[self.n_input_tokens:],
        #                 self.output_text,
        #                 len(self.outputs),
        #                 ', '.join(["{}:'{}'".format(idx, t) for idx, t in enumerate(self.tokens)]))

    def plot_feature_importance_barplots(self):
        """
        Barplot showing the improtance of each input token. Prints one barplot
        for each generated token.
        TODO: This should be LMOutput I think
        :return:
        """
        printable_tokens = [repr(token) for token in self.tokens]
        for i in self.importance:
            importance = i.numpy()
            lm_plots.token_barplot(printable_tokens, importance)
            # print(i.numpy())
            plt.show()

    # def plot_inner_predictions(self, lmhead):
    #     # Take the hidden state from each output
    #     hidden_states = self.hidden_states
    #     # n_layers = len(self.outputs[0][-2]) - 1 # number of layers (excluding inputs)
    #     # hidden_states = np.zeros((n_layers,len(self.outputs))) # (n_layers X # of generation steps (tokens in output seq)
    #     # for output_position_id, output in enumerate(self.outputs):
    #     #     step_hidden_states = output[-2][1:]
    #     #     hidden_states[:, output_position_id] = [hs[-1] for hs in step_hidden_states]
    #
    #     # layers X steps
    #     n_layers, position = len(hidden_states), hidden_states[0].shape[0]
    #
    #     predicted_tokens = np.empty((n_layers - 1, position), dtype='U25')
    #     softmax_scores = np.zeros((n_layers - 1, position))
    #     token_found_mask = np.ones((n_layers - 1, position))
    #
    #     for i, level in enumerate(hidden_states[1:]):  # loop through layer levels
    #         for j, logits in enumerate(level):  # Loop through positions
    #             scores = lmhead(logits)
    #             sm = F.softmax(scores, dim=-1)
    #             token_id = torch.argmax(sm)
    #             token = self.tokenizer.decode([token_id])
    #             predicted_tokens[i, j] = token
    #             softmax_scores[i, j] = sm[token_id]
    #             #         print('layer', i, 'position', j, 'top1', token_id, 'actual label', output['token_ids'][j]+1)
    #             if token_id == self.token_ids[j + 1]:
    #                 token_found_mask[i, j] = 0
    #
    #     lm_plots.plot_logit_lens(self.tokens, softmax_scores, predicted_tokens,
    #                              token_found_mask=token_found_mask,
    #                              show_input_tokens=False,
    #                              n_input_tokens=self.n_input_tokens
    #                              )

    def layer_predictions(self, position: int = 0, topk: Optional[int] = 10, layer: Optional[int] = None, **kwargs):
        """
        Visualization plotting the topk predicted tokens after each layer (using its hidden state).
        :param output: OutputSeq object generated by LM.generate()
        :param position: The index of the output token to trace
        :param topk: Number of tokens to show for each layer
        :param layer: None shows all layers. Can also pass an int with the layer id to show only that layer
        """

        watch = self.to(torch.tensor([self.token_ids[self.n_input_tokens]]))
        # There is one lm output per generated token. To get the index
        output_index = position - self.n_input_tokens
        if layer is not None:
            hidden_states = self.hidden_states[layer + 1].unsqueeze(0)
        else:
            hidden_states = self.hidden_states[1:]  # Ignore the first element (embedding)

        k = topk
        top_tokens = []
        probs = []
        data = []

        print('Predictions for position {}'.format(position))
        for layer_no, h in enumerate(hidden_states):
            # print(h.shape)
            hidden_state = h[position - 1]
            # Use lm_head to project the layer's hidden state to output vocabulary
            logits = self.lm_head(hidden_state)
            softmax = F.softmax(logits, dim=-1)
            sorted_softmax = self.to(torch.argsort(softmax))

            # Not currently used. If we're "watching" a specific token, this gets its ranking
            # idx = sorted_softmax.shape[0] - torch.nonzero((sorted_softmax == watch)).flatten()

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
        """.format(viz_id, viz_id, json.dumps(data))
        d.display(d.Javascript(js))

        if 'printJson' in kwargs and kwargs['printJson']:
            print(data)
            return data

    def rankings(self, **kwargs):
        """
        Plots the rankings (across layers) of the tokens the model selected.
        Each column is a position in the sequence. Each row is a layer.
        """
        hidden_states = self.hidden_states

        n_layers = len(hidden_states)
        position = hidden_states[0].shape[0] - self.n_input_tokens + 1
        # print('position', position)

        predicted_tokens = np.empty((n_layers - 1, position), dtype='U25')
        rankings = np.zeros((n_layers - 1, position), dtype=np.int32)
        token_found_mask = np.ones((n_layers - 1, position))

        # loop through layer levels
        for i, level in enumerate(hidden_states[1:]):
            # Loop through generated/output positions
            for j, hidden_state in enumerate(level[self.n_input_tokens - 1:]):
                # print('hidden state layer', i, 'position', self.n_input_tokens-1+j)
                # Project hidden state to vocabulary
                # (after debugging pain: ensure input is on GPU, if appropriate)
                logits = self.lm_head(hidden_state)
                # logits = self.lm_head(torch.tensor(hidden_state))
                # Sort by score (ascending)
                sorted = torch.argsort(logits)
                # What token was sampled in this position?
                token_id = torch.tensor(self.token_ids[self.n_input_tokens + j])
                # print('token_id', token_id)
                # What's the index of the sampled token in the sorted list?
                r = torch.nonzero((sorted == token_id)).flatten()
                # subtract to get ranking (where 1 is the top scoring, because sorting was in ascending order)
                ranking = sorted.shape[0] - r
                # print('ranking', ranking)
                # token_id = torch.argmax(sm)
                token = self.tokenizer.decode([token_id])
                predicted_tokens[i, j] = token
                rankings[i, j] = int(ranking)
                #  print('layer', i, 'position', j, 'top1', token_id, 'actual label', output['token_ids'][j]+1)
                if token_id == self.token_ids[j + 1]:
                    token_found_mask[i, j] = 0

        input_tokens = [repr(t) for t in self.tokens[self.n_input_tokens - 1:-1]]
        output_tokens = [repr(t) for t in self.tokens[self.n_input_tokens:]]
        # print('in out', input_tokens, output_tokens)
        lm_plots.plot_inner_token_rankings(input_tokens,
                                           output_tokens,
                                           rankings,
                                           **kwargs)

        if 'printJson' in kwargs and kwargs['printJson']:
            data = {'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'rankings': rankings,
                    'predicted_tokens': predicted_tokens}
            print(data)
            return data

    def rankings_watch(self, watch: List[int] = None, position: int = -1, **kwargs):
        """
        Plots the rankings of the tokens whose ids are supplied in the watch list.
        Only considers one position.
        """
        if position != -1:
            position = position - 1  # e.g. position 5 corresponds to activation 4

        hidden_states = self.hidden_states

        n_layers = len(hidden_states)
        n_tokens_to_watch = len(watch)

        # predicted_tokens = np.empty((n_layers - 1, n_tokens_to_watch), dtype='U25')
        rankings = np.zeros((n_layers - 1, n_tokens_to_watch), dtype=np.int32)

        # loop through layer levels
        for i, level in enumerate(hidden_states[1:]):  # Skip the embedding layer
            # Loop through generated/output positions
            for j, token_id in enumerate(watch):
                hidden_state = level[position]
                # Project hidden state to vocabulary
                # (after debugging pain: ensure input is on GPU, if appropriate)
                logits = self.lm_head(hidden_state)
                # logits = lmhead(torch.tensor(hidden_state))
                # Sort by score (ascending)
                sorted = torch.argsort(logits)
                # What token was sampled in this position?
                token_id = torch.tensor(token_id)
                # print('token_id', token_id)
                # What's the index of the sampled token in the sorted list?
                r = torch.nonzero((sorted == token_id)).flatten()
                # subtract to get ranking (where 1 is the top scoring, because sorting was in ascending order)
                ranking = sorted.shape[0] - r
                # print('ranking', ranking)
                # token_id = torch.argmax(sm)
                # token = self.tokenizer.decode([token_id])
                # predicted_tokens[i, j] = token
                rankings[i, j] = int(ranking)
                #  print('layer', i, 'position', j, 'top1', token_id, 'actual label', output['token_ids'][j]+1)
                # if token_id == self.token_ids[j + 1]:
                #     token_found_mask[i, j] = 0

        input_tokens = [t for t in self.tokens]
        output_tokens = [repr(self.tokenizer.decode(t)) for t in watch]
        # print('in out', input_tokens, output_tokens)
        lm_plots.plot_inner_token_rankings_watch(input_tokens,
                                                 output_tokens,
                                                 rankings)

        if 'printJson' in kwargs and kwargs['printJson']:
            data = {'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'rankings': rankings}
            print(data)
            return data

    def run_nmf(self, **kwargs):
        """
        Run Non-negative Matrix Factorization on network activations of FFNN.
        Saves the components in self.components

        """
        return NMF(self.activations,
                   n_input_tokens=self.n_input_tokens,
                   token_ids=self.token_ids,
                   _path=self._path,
                   tokens=self.tokens, **kwargs)

    def attention(self, attention_values=None, layer=0, **kwargs):

        position = self.n_input_tokens

        # importance_id = position - self.n_input_tokens

        importance_id = self.n_input_tokens - 1  # Sete first values to first output token
        tokens = []
        if attention_values:
            attn = attention_values
        else:

            attn = self.attention_values[layer]
            # normalize attention heads
            attn = attn.sum(axis=1) / attn.shape[1]

        for idx, token in enumerate(self.tokens):
            # print(idx, attn.shape)
            type = "input" if idx < self.n_input_tokens else 'output'
            if idx < len(attn[0][importance_id]):
                attention_value = attn[0][importance_id][idx].cpu().detach().numpy()
            else:
                attention_value = 0

            tokens.append({'token': token,
                           'token_id': int(self.token_ids[idx]),
                           'type': type,
                           'value': str(attention_value),  # because json complains of floats
                           'position': idx
                           })

        data = {
            'tokens': tokens,
            'attributions': [att.tolist() for att in attn[0].cpu().detach().numpy()]
        }

        d.display(d.HTML(filename=os.path.join(self._path, "html", "setup.html")))
        d.display(d.HTML(filename=os.path.join(self._path, "html", "basic.html")))
        viz_id = 'viz_{}'.format(round(random.random() * 1000000))
        js = """
         requirejs(['basic', 'ecco'], function(basic, ecco){{
            const viz_id = basic.init()

            ecco.interactiveTokens(viz_id, {})
         }}, function (err) {{
            console.log(err);
        }})""".format(data)
        d.display(d.Javascript(js))

        if 'printJson' in kwargs and kwargs['printJson']:
            print(data)


class NMF:
    " Conducts NMF and holds the models and components "

    def __init__(self, activations: np.ndarray,
                 n_input_tokens: int = 0,
                 token_ids: torch.Tensor = torch.Tensor(0),
                 _path: str = '',
                 n_components: int = 10,
                 from_layer: Optional[int] = None,
                 to_layer: Optional[int] = None,
                 tokens: Optional[List[str]] = None, **kwargs):
        self._path = _path
        self.token_ids = token_ids
        self.n_input_tokens = n_input_tokens

        if len(activations.shape) != 3:
            raise ValueError(f"The 'activations' parameter should have three dimensions: (layers, neurons, positions). "
                             f"Supplied dimensions: {activations.shape}", 'activations')

        if from_layer is not None or to_layer is not None:
            from_layer = from_layer if from_layer is not None else 0
            to_layer = to_layer if to_layer is not None else activations.shape[0]

            if from_layer == to_layer:
                raise ValueError(f"from_layer ({from_layer}) and to_layer ({to_layer}) cannot be the same value. "
                                 "They must be apart by at least one to allow for a layer of activations.")

            if from_layer > to_layer:
                raise ValueError(f"from_layer ({from_layer}) cannot be larger than to_layer ({to_layer}).")

            merged_act = np.concatenate(activations[from_layer: to_layer], axis=0)
            activations = np.expand_dims(merged_act, axis=0)

        self.tokens = tokens
        " Run NMF. Activations is neuron activations shaped (layers, neurons, positions)"
        n_output_tokens = activations.shape[-1]
        n_layers = activations.shape[0]
        n_components = min([n_components, n_output_tokens])
        components = np.zeros((n_layers, n_components, n_output_tokens))
        models = []

        # Get rid of negative activation values
        # (There are some, because GPT2 uses GLEU, which allow small negative values)
        activations = np.maximum(activations, 0)

        for idx, layer in enumerate(activations):
            #     print(layer.shape)
            model = decomposition.NMF(n_components=n_components,
                                      init='random',
                                      random_state=0,
                                      max_iter=500,
                                      **kwargs)
            components[idx] = model.fit_transform(layer.T).T
            models.append(model)

        self.models = models
        self.components = components

    def explore(self, **kwargs):
        # position = self.n_input_tokens + 1

        tokens = []
        for idx, token in enumerate(self.tokens):  # self.tokens[:-1]
            type = "input" if idx < self.n_input_tokens else 'output'
            tokens.append({'token': token,
                           'token_id': int(self.token_ids[idx]),
                           'type': type,
                           # 'value': str(components[0][comp_num][idx]),  # because json complains of floats
                           'position': idx
                           })

        # Duplicate the factor at index 'n_input_tokens'. THis way
        # each token has an activation value (instead of having one activation less than tokens)
        # But with different meanings: For inputs, the activation is a response
        # For outputs, the activation is a cause
        # print('shape', components.shape)
        # for i, comp in enumerate(components[0]):
        #     print(i, comp, '\nconcat:', np.concatenate([comp[:self.n_input_tokens], comp[self.n_input_tokens-1:]]))
        factors = np.array(
            [[np.concatenate([comp[:self.n_input_tokens], comp[self.n_input_tokens - 1:]]) for comp in
              self.components[0]]])
        factors = [comp.tolist() for comp in factors]

        data = {
            'tokens': tokens,
            'factors': factors
        }

        d.display(d.HTML(filename=os.path.join(self._path, "html", "setup.html")))
        d.display(d.HTML(filename=os.path.join(self._path, "html", "basic.html")))
        viz_id = 'viz_{}'.format(round(random.random() * 1000000))
        # print(data)
        js = """
         requirejs(['basic', 'ecco'], function(basic, ecco){{
            const viz_id = basic.init()
            ecco.interactiveTokensAndFactorSparklines(viz_id, {})
         }}, function (err) {{
            console.log(err);
        }})""".format(data)
        d.display(d.Javascript(js))

        if 'printJson' in kwargs and kwargs['printJson']:
            print(data)

    def plot(self, n_components=3):

        for idx, comp in enumerate(self.components):
            #     print('Layer {} components'.format(idx), 'Variance: {}'.format(lm.variances[idx][:n_components]))
            print('Layer {} components'.format(idx))
            comp = comp[:n_components, :].T

            #     plt.figure(figsize=(16,2))
            fig, ax1 = plt.subplots(1)
            plt.subplots_adjust(wspace=.4)
            fig.set_figheight(2)
            fig.set_figwidth(17)
            #     fig.tight_layout()
            # PCA Line plot
            ax1.plot(comp)
            ax1.set_xticks(range(len(self.tokens)))
            ax1.set_xticklabels(self.tokens, rotation=-90)
            ax1.legend(['Component {}'.format(i + 1) for i in range(n_components)], loc='center left',
                       bbox_to_anchor=(1.01, 0.5))

            plt.show()

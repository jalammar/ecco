from ecco import lm_plots
from ecco.language_model import LM
from sklearn.decomposition import PCA
from ecco.output import OutputSeq
import torch
import IPython.display as d
import os
import transformers
from typing import Optional
import numpy as np
import json
from sklearn.mixture import GaussianMixture
from sklearn import cluster
import seaborn as sns
import matplotlib.pyplot as plt
from torch.nn import functional as F
import random

from transformers import AutoTokenizer, AutoModelForCausalLM


class LMActivations(LM):
    def __init__(self, model, tokenizer, collect_activations_flag=False,
                 collect_gen_activations_flag=False):
        super().__init__(model, tokenizer)

        self._reset()
        self._hooks = {}

        self.components = None
        self.variances = None
        self.importances = None
        self.final_model_output = None
        self.topk_importances = []
        self.input_ids = None
        self.neurons_to_inhibit = {}
        self.neurons_to_induce = {}

        self.collect_activations_flag = collect_activations_flag
        self.collect_gen_activations_flag = collect_gen_activations_flag

        self._attach_hooks(self.model)

        self.device = 'cuda' if torch.cuda.is_available() and self.model.device.type == 'cuda' \
            else 'cpu'

    def _reset(self):
        self._all_activations_dict = {}
        self._generation_activations_dict = {}
        self._input_token_ids = []
        self.activations = []
        self.all_activations = []
        self.generation_activations = []
        self.tokens = []
        self.cluster_neurons_dict = None
        self.clustering_layer = -1

    def _attach_hooks(self, model):
        for name, module in model.named_modules():
            # print(name)
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
                self._hooks[name+'_inhibit'] = module.register_forward_pre_hook(
                    lambda self_, input_, name=name: \
                    self._inhibit_neurons_hook(name, input_)
                )

            elif isinstance(model, transformers.modeling_reformer.ReformerModelWithLMHead):
                if "feed_forward.output" in name and "dense" not in name:
                    self._hooks[name] = module.register_forward_hook(
                        lambda self_, input_, output,
                               name=name: self._get_activations_hook(name, input_, output))




    def _get_activations_hook(self, name, input_):
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

    def _get_generation_activations_hook(self, name, input_):
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


    def _inhibit_neurons_hook(self, name, input_tensor):
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
                input_tensor[0][0][-1][n] = input_tensor[0][0][-1][n] * 10 # tuple, batch, position

        return input_tensor

    def generate_fi(self, input_str: str, max_length: Optional[int] = 128,
                    temperature=None, top_k=None, top_p=None, do_sample=None,
                    generate=None):
        """
        Generate with Feature Importance.
        Generates text token by token without using the model's generate method. This provides
        more ability to control the generation process to be able to do things like:
        - Trace the gradient backwards to gauge feature importance
        - More control over sampling

        :return:
        """
        # self.activations.to('cuda:0')

        self._reset()

        top_k = top_k if top_k is not None else self.model.config.top_k
        top_p = top_p if top_p is not None else self.model.config.top_p
        temperature = temperature if temperature is not None else self.model.config.temperature
        do_sample = do_sample if do_sample is not None else self.model.config.task_specific_params['text-generation'][
            'do_sample']

        embedding_matrix = self.model.transformer.wte.weight

        input_ids = self.tokenizer(input_str, return_tensors="pt")['input_ids'][0]
        self.input_ids = input_ids
        n_input_tokens = len(input_ids)

        if generate is not None:
            max_length = n_input_tokens + generate

        past = None
        outputs = []
        importances = []
        cur_len = len(input_ids)

        assert cur_len < max_length, \
            "max_length set to {} while input token has more tokens ({}). Consider increasing max_length" \
                .format(max_length, cur_len)

        while cur_len < max_length:
            output_token_id, importance, output, past = self._generate_token(input_ids,
                                                                             past,  # Note, this is not currently
                                                                             # used because attribution needs the entire sequence
                                                                             embedding_matrix,
                                                                             temperature=temperature,
                                                                             top_k=top_k, top_p=top_p,
                                                                             do_sample=do_sample)
            # outputs=output
            input_ids = torch.cat([input_ids, torch.tensor([output_token_id])])
            importances.append(importance.detach().cpu().numpy())
            cur_len = cur_len + 1
            if output_token_id == self.model.config.eos_token_id:
                break
        output_token_ids = input_ids
        self.importances = importances
        hidden_states = output[2]

        # # Save tokens
        # self.tokens = []
        # for i in output_token_ids:
        #     token = self.tokenizer.decode([i])
        #     self.tokens.append(token)

        tokens = []
        for i in input_ids:
            token = self.tokenizer.decode([i])
            tokens.append(token)
        self.tokens = tokens

        return OutputSeq(**{'tokenizer': self.tokenizer,
                            'token_ids': input_ids,
                            'n_input_tokens': n_input_tokens,
                            'output_text': self.tokenizer.decode(input_ids),
                            'tokens': tokens,
                            'hidden_states': hidden_states,
                            'attribution': importances,
                            'activations': self.activations})
        # return {'token_ids': input_ids,
        #         'output_text': self.tokenizer.decode(input_ids)}

    def _generate_token(self, input_ids, past, embedding_matrix,
                        do_sample, temperature, top_k, top_p):
        """
        Generate a token. Gets the embeddings of tokens instead of token ids so feature importance can
        be calculated.
        :param input_ids:
        :param past:
        :param embedding_matrix:
        :param do_sample:
        :param temperature:
        :param top_k:
        :param top_p:
        :return:
        """
        vocab_size = embedding_matrix.shape[0]
        one_hot_tensor = self._one_hot(input_ids, vocab_size)
        # token_ids_tensor_one_hot = torch.tensor(one_hot_tensor, requires_grad=True)
        token_ids_tensor_one_hot = one_hot_tensor.clone().requires_grad_(True)
        token_ids_tensor_one_hot = token_ids_tensor_one_hot.to(self.device)
        token_ids_tensor_one_hot.retain_grad()
        # embedding_matrix = embedding_matrix.to(self.device)

        # print('token_ids_tensor_one_hot', token_ids_tensor_one_hot.device, token_ids_tensor_one_hot.requires_grad_())
        inputs_embeds = torch.matmul(token_ids_tensor_one_hot, embedding_matrix)
        inputs_embeds = inputs_embeds.to(self.device)

        # print(one_hot_tensor.device, inputs_embeds.device, embedding_matrix.device, token_ids_tensor_one_hot.device)

        # ========= model
        # Pay attention if using 'past' to only pass in the embeddings of the current token,
        # not of all previous ones as well. But in that case, will we be able to calculate the gradient
        # Back to the input tokens? I doubt it because the graph now starts with only one token?
        # It'll depend on how the past is represented in the ghraph I think.
        # Decision: I'll keep it. I will not use past.
        output = self.model(inputs_embeds=inputs_embeds, return_dict=True)
        predict = output[0]
        past = output['past_key_values']
        self.final_model_output = output

        next_token_logits = predict[-1, :]
        scores = next_token_logits
        # print(torch.topk(predict[inputs_embeds.shape[0] - 1], 2))

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
            prediction_id = torch.argmax(next_token_logits, dim=-1)

        # prediction_id = torch.argmax(predict[inputs_embeds.shape[0] - 1])
        # print(self.tokenizer.decode([prediction_id]))

        # prediction_id now has the id of the token we want to output
        # To do feature importance, let's get the actual logit associated with
        # this token
        prediction_logit = predict[inputs_embeds.shape[0] - 1][prediction_id]

        prediction_logit.backward(retain_graph=True)
        # print(token_ids_tensor_one_hot.device, token_ids_tensor_one_hot.requires_grad_())
        token_importance_raw = torch.norm(token_ids_tensor_one_hot.grad, dim=1)
        token_importance = token_importance_raw / torch.sum(token_importance_raw)
        # print(token_importance)

        token_ids_tensor_one_hot.grad.data.zero_()

        return prediction_id, token_importance, output, past

    def run_pca(self, layer=None, n_components=40):
        """
        Runs PCA on all layers, saves the components in self.components
        saves variences in self.variances
        """

        n_output_tokens = self.activations.shape[2]
        n_layers = self.activations.shape[0]
        n_components = min([n_components, n_output_tokens])  # len(self.tokens)

        activations = self.activations if layer == None else [self.activations[layer]]
        components = np.zeros((n_layers, n_components, n_output_tokens))
        variances = np.zeros((n_layers, n_components))
        # n_components = min([n_components, len(self.tokens)])
        for idx, layer in enumerate(activations):
            pca = PCA(n_components=n_components)
            reduced = pca.fit_transform(layer.T).T
            variances[idx] = pca.explained_variance_ratio_
            components[idx] = reduced

        self.components = components
        self.variances = variances

    def explore_pca(self, layer=0, component=0):

        params = {'tokens': self.tokens,
                  'activations': self.components.tolist(),
                  'layer': layer,
                  'variances': self.variances.tolist()
                  }

        viz_id = 'viz_{}'.format(round(random.random() * 1000000))
        # d.display(d.Javascript('window.params["{}"]={}'.format(viz_id, json.dumps(params))))

        d.display(d.HTML(filename=os.path.join(self._path, "html", "setup.html")))

        d.display(d.HTML(filename=os.path.join(self._path, "html", "pca_explorable.html")))
        js = """
        requirejs(['pca_explorable'], function(pca_explorable){{
        if (window.pca === undefined)
            window.pca = {{}}
        window.pca["{}"] = new pca_explorable.PCA_Explorable("{}", {}, {})
        }}
        )
        """.format(viz_id, viz_id, json.dumps(params), component)
        d.display(d.Javascript(js))

    def explore_layer_neurons(self, layer=0):
        """
        The most basic neuron explorable. Tokens of all the neurons in a given layer
        """
        params = {'tokens': self.tokens,
                  'activations': self.activations[layer].tolist(),
                  'layer': layer}

        d.display(d.Javascript('window.params={}'.format(json.dumps(params))))
        d.display(d.HTML(filename=os.path.join(self._path, "html", "neuron_explorable.html")))
        # d.display(d.Javascript(filename=os.path.join(self._path, "./NeuronVizCluster.js")))

    # Alias. DELETE LATER
    def show_text_activations(self, layer=0):
        return self.explore_layer_neurons(self, layer)

    def attention(self):
        d.display(d.HTML(filename=os.path.join(self._path, "html", "Ecco-Attention.html")))

    def explore_all_layers(self):
        """
        For each layer in the network, shows the token, highlighted by the sum of activation values.

        :return:
        """
        positive_activations = np.maximum(0, self.activations)
        per_layer_activations_sum = np.sum(positive_activations, axis=1)
        # (layers, tokens)
        params = {'tokens': self.tokens,
                  'per_layer_activations_sum': per_layer_activations_sum.tolist()}

        d.display(d.Javascript('window.params={}'.format(json.dumps(params))))
        d.display(d.HTML(filename=os.path.join(self._path, "html", "layer_activations.html")))

    # Alias. DELETE LATER
    def layer_activations(self):
        return self.layer_activations()

    def explore_layer_activations(self, layer=0, sort="n_active_neurons_threshold", threshold=1):
        """
        Explorable showing interesting neurons.
        Interesting as sorted by number of activations above threshold

        :param layer:
        :param sort:
        :param threshold:
        :return:
        """
        activations = self.activations[layer]
        sorted_indices = np.arange(activations.shape[0])
        if sort == "n_active_neurons_threshold":
            # For each neuron, count how many tokens activated it above threshold
            n_active_neurons_per_token = np.count_nonzero(activations > threshold, axis=1)
            # Sort the neurons by how many tokens got them above activation threshold
            sorted_indices = np.argsort(n_active_neurons_per_token)[::-1]
            # Re-arrange activation according to this sort
            sorted_activations = np.take(activations, sorted_indices, axis=0)

            activations = sorted_activations

        params = {'tokens': self.tokens,
                  'activations': activations.tolist(),
                  'neuron_ids': sorted_indices.tolist(),
                  'layer': layer}

        d.display(d.Javascript('window.params={}'.format(json.dumps(params))))
        d.display(d.HTML(filename=os.path.join(self._path, "html", "neuron_explorable.html")))

    # Alias. DELETE LATER
    def full_layer_activations(self, layer=0, sort="n_active_neurons_threshold", threshold=1):
        return self.explore_layer_activations(layer, sort, threshold)

    def cluster_activations(self, layer, algorithm="kmeans", n_clusters=30, **kwargs):
        kws = {} if kwargs is None else kwargs.copy()

        if algorithm == "GaussianMixture":
            kws['n_components'] = n_clusters
            algorithm = GaussianMixture(**kws)
        elif algorithm == "DBSCAN":
            algorithm = cluster.DBSCAN(**kws)
        elif 'cluster_obj' in kws:
            algorithm = kws['cluster_obj']
        else:
            kws['n_clusters'] = n_clusters
            algorithm = cluster.KMeans(**kws)

        pred = algorithm.fit_predict(self.activations[layer])

        # build a dict clusters:
        # { cluster_id : [neuron ids of neurons in this cluster] }
        self.cluster_neurons_dict = {}
        for idx, n in enumerate(pred):
            if n not in self.cluster_neurons_dict:
                self.cluster_neurons_dict[n] = [idx]
            else:
                self.cluster_neurons_dict[n].append(idx)

        self.clustering_layer = layer

    def explore_all_clusters(self, min_neurons=3):
        """
        Explorable showing the generated text highlighted by the average of each cluster's activations.
        + Buttons to cycle between clusters.

        TODO:
        - Adapt for character-level tokens
        - Show number of neurons in clusters
        - Change UI to indicate we're browsing through clusters, not neurons
        - WISHLIST: On hover over cluster row, show tooltip showing a heatmap of cluster activations

        :param min_neurons:
        :return:
        """

        assert self.cluster_neurons_dict is not None, \
            "No clustering found. Make sure to cluster activations using cluster_activations()"

        averages = []
        cluster_ids = []
        ignored_clusters = []
        n_neurons_per_cluster = []
        for cluster_id, neuron_ids in self.cluster_neurons_dict.items():
            if len(neuron_ids) > min_neurons:
                cluster_activations = self.activations[self.clustering_layer][neuron_ids]
                averages.append(np.mean(cluster_activations, axis=0).tolist())
                n_neurons_per_cluster.append(len(neuron_ids))
            else:
                ignored_clusters.append(cluster_id)

        params = {'tokens': self.tokens,
                  'activations': averages,
                  'neuron_ids': cluster_ids,
                  'layer': self.clustering_layer,
                  'n_neurons_per_cluster': n_neurons_per_cluster
                  }

        d.display(d.Javascript('window.params={}'.format(json.dumps(params))))
        d.display(d.HTML(filename=os.path.join(self._path, "html", "cluster_explorable.html")))

        if len(ignored_clusters) > 0:
            print('ignored clusters {} for having less than {} neurons'
                  .format(ignored_clusters, min_neurons))

    def explore_cluster(self, cluster_id):
        """
        Highlighted generated text + heatmap for each neuron inside the cluster.

        TODO:
        - Show neuron numbers in the heatmap
        - When a row is clicked, switch to that neuron
        - Highlight rows based on active neuron


        :param cluster_id:
        :return:
        """

        neuron_ids = self.cluster_neurons_dict[cluster_id]
        activations = self.activations[self.clustering_layer][neuron_ids]
        params = {'tokens': self.tokens,
                  'activations': activations.tolist(),
                  'neuron_ids': neuron_ids,
                  'layer': self.clustering_layer}

        d.display(d.Javascript('window.params={}'.format(json.dumps(params))))
        d.display(d.HTML(filename=os.path.join(self._path, "html", "neuron_explorable.html")))

    # =====================================================
    # =====================================================
    # Plots

    def plot_cluster_activations(self, cluster_id, height=20):

        neuron_ids = self.cluster_neurons_dict[cluster_id]
        activations = self.activations[self.clustering_layer][neuron_ids]

        lm_plots.plot_activations(self.tokens[1:], activations, height=height)

    def plot_all_clusters(self):

        assert self.cluster_neurons_dict is not None, \
            "No clustering found. Make sure to cluster activations using cluster_activations()"

        activations = None
        cluster_ids = []
        for cluster_id, neuron_ids in self.cluster_neurons_dict.items():
            cluster_ids.append(cluster_id)
            if activations is None:  # Add the first element
                activations = self.activations[self.clustering_layer][neuron_ids]
            else:  # Concat the next elements
                cluster_activations = self.activations[self.clustering_layer][neuron_ids]
                activations = np.concatenate([activations, cluster_activations], axis=0)

        lm_plots.plot_clustered_activations(self.tokens[1:],
                                            np.array(activations),
                                            self.cluster_neurons_dict,
                                            cluster_ids)

    def plot_layer_activations(self, layer):
        self.plot_activations(self.tokens[1:], self.activations[layer])

    def plot_activations(self, tokens, activations, file_prefix='neuron_activation_plot'):
        return lm_plots.plot_activations(tokens, activations, file_prefix=file_prefix)
        """ Plots a heat mapshowing how active each neuron (row) was with each token
        (columns). Neurons with activation less then masking_threashold are masked.

        Args:
          tokens: list of the tokens. Note if you're examining activations
          associated with the token as input or as output.


        """

    def plot_feature_importance_barplots(self):
        """
        Barplot showing the improtance of each input token. Prints one barplot
        for each generated token.
        TODO: This should be LMOutput I think
        :return:
        """
        printable_tokens = [repr(token) for token in self.tokens]
        for i in self.importances:
            importance = i.numpy()
            lm_plots.token_barplot(printable_tokens, importance)
            # print(i.numpy())
            plt.show()

    def _one_hot(self, token_ids, vocab_size):
        return torch.zeros(len(token_ids), vocab_size).scatter_(1, token_ids.unsqueeze(1), 1.)

    def plot_inner_predictions(self, output):
        hidden_states = self.final_model_output[-2]

        n_levels, position = len(hidden_states), hidden_states[0].shape[0]

        predicted_tokens = np.empty((n_levels - 1, position), dtype='U25')
        softmax_scores = np.zeros((n_levels - 1, position))
        token_found_mask = np.ones((n_levels - 1, position))

        for i, level in enumerate(hidden_states[1:]):  # loop through layer levels
            for j, logits in enumerate(level):  # Loop through positions
                scores = self.model.lm_head(logits)
                sm = F.softmax(scores, dim=-1)
                token_id = torch.argmax(sm)
                token = self.tokenizer.decode([token_id])
                predicted_tokens[i, j] = token
                softmax_scores[i, j] = sm[token_id]
                #         print('layer', i, 'position', j, 'top1', token_id, 'actual label', output['token_ids'][j]+1)
                if token_id == output['token_ids'][j + 1]:
                    token_found_mask[i, j] = 0

        lm_plots.plot_logit_lens(self.tokens, softmax_scores, predicted_tokens,
                                 token_found_mask=token_found_mask,
                                 show_input_tokens=False,
                                 n_input_tokens=len(self.input_ids) - 1
                                 )

import torch
import numpy as np


def reshape_hidden_states_to_3d(hidden_states):
    """
    Turn hidden_states from (layer, batch, position, d_model)
    to a tensor  (layer, d_model, batch + position).
    Args:
        hidden_states: the hidden states return by the language model. A list of tensors. Its shape:
            (layer, batch, position, d_model)
    returns:
        hidden_states: tensor in the shape (layer, d_model, batch + position)
    """
    hs = hidden_states

    # Turn from a list of tensors into a tensor
    if isinstance(hs, tuple):
        hs = torch.stack(hs)

    # Merge the batch and position dimensions
    hs = hs.reshape((hs.shape[0], -1, hs.shape[-1]))

    return hs


def reshape_activations_to_3d(activations):
    """
    Reshape the activations tensors into a shape where it's easier to compare
    activation vectors.
    Args:
        activations: activations tensor of LM. Shape:
            (batch, layer, neuron, position)
    returns:
        activations: activations tensor reshaped into:
            (layer, neuron, batch + position)
    """

    # Swap axes from (0 batch, 1 layer, 2 neuron, 3 position)
    # to (0 layer, 1 neuron, 2 batch, 3 position)
    activations = np.moveaxis(activations, [0, 1, 2], [2, 0, 1])
    s = activations.shape
    acts = activations.reshape(s[0], s[1], -1)
    return acts




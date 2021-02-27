import torch


def reshape_hidden_states_to_3d(hidden_states):
    """
    Turn hidden_states from (layer, batch, position, d_model)
    to a tensor  (layer, d_model, batch + position).
    Args:
        hidden_states: the hidden states return by the language model. A list of tensors. Its shape:
            (layer, batch, position, d_model)
      """
    hs = hidden_states

    # Turn from a list of tensors into a tensor
    if isinstance(hs, tuple):
        hidden_states = torch.stack(hs)

    # Merge the batch and position dimensions
    hs = hs.reshape((hs.shape[0], -1, hs.shape[-1]))
    return hs

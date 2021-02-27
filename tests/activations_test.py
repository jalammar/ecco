import pytest
import torch
import numpy as np
from ecco import activations


class TestActivations:
    def test_reshape_hidden_states(self):
        # Create a tensor of shape (1,2,3,1)
        t = torch.stack([torch.ones(3), torch.zeros(3)]).unsqueeze(0).unsqueeze(-1)
        result = activations.reshape_hidden_states_to_3d(t)
        assert result.shape == (1, 6, 1)
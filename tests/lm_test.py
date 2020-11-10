
from ecco import output
from ecco.language_model.lm import _one_hot, LM
import pytest
import torch


class TestLM:
    def test_one_hot(self):
        expected = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        actual = _one_hot(torch.tensor([0, 1]), 3)
        assert torch.all(torch.eq(expected, actual))


    def test_generate_(self):
        pass

    def test_activations_dict_to_array(self, act_dict):
        pass

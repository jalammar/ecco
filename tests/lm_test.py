
from ecco import output
from ecco.language_model.lm import _one_hot, LM, sample_output_token, activations_dict_to_array
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
from unittest.mock import patch
from transformers import AutoTokenizer, AutoModelForCausalLM

# @pytest.fixture
# def mockLM():
#     #setup
#     class mockLM
#     #yield
#
#     #teardown


class TestLM:
    def test_one_hot(self):
        expected = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        actual = _one_hot(torch.tensor([0, 1]), 3)
        assert torch.all(torch.eq(expected, actual))

    def test_select_output_token_argmax(self):
        result = sample_output_token(torch.tensor([0., 1.]), False, 0, 0, 0)
        assert result == torch.tensor(1)

    def test_select_output_token_sample(self):
        result = sample_output_token(torch.tensor([0., 0.5, 1.]), True, 1, 1, 1)
        assert result == torch.tensor(2)

    def test_activations_dict_to_array(self):
        dict = {0:[[np.zeros((3,4))]],
                1:[[np.zeros((3,4))]]}
        activations = activations_dict_to_array(dict)
        assert activations.shape == (2,4,3)

    # def test_generate_token_no_attribution(self, mocker):
    #     pass
    #
    # def test_generate_token_with_attribution(self, mocker):
    #     pass

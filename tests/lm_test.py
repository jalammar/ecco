
from ecco import output
from ecco.language_model.lm import _one_hot, LM
import pytest
import torch
from unittest.mock import MagicMock
from unittest.mock import patch
from transformers import AutoTokenizer, AutoModelForCausalLM

@pytest.fixture
def mockLM():
    #setup
    class mockLM
    #yield

    #teardown


class TestLM:
    def test_one_hot(self):
        expected = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        actual = _one_hot(torch.tensor([0, 1]), 3)
        assert torch.all(torch.eq(expected, actual))


    def test_generate_token_no_attribution(self):

        # set up -- mock model
        lm = LM(model, tokenizer)

        pred_id, output, past = lm._generate_token([0], None, 1,0, 50, 1)

    def test_select_output_token(self):
        pass

    def test_activations_dict_to_array(self, act_dict):
        pass

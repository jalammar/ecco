from ecco.lm import LM, _one_hot, sample_output_token, activations_dict_to_array
import ecco
import torch
import numpy as np
from transformers import PreTrainedModel 


class TestLM:
    def test_one_hot(self):
        expected = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        actual = _one_hot(torch.tensor([0, 1]), 3)
        assert torch.all(torch.eq(expected, actual))

    def test_select_output_token_argmax(self):
        result = sample_output_token(torch.tensor([0., 1.]), False, 0, 0, 0)
        assert result == torch.tensor(1)

    def test_select_output_token_sample(self):
        result = sample_output_token(torch.tensor([[0., 0.5, 1.]]), True, 1, 1, 1.0)
        assert result == torch.tensor(2)

    def test_activations_dict_to_array(self):
        batch, position, neurons = 1, 3, 4
        actual_dict = {0: np.zeros((batch, position, neurons)),
                       1: np.zeros((batch, position, neurons))}
        activations = activations_dict_to_array(actual_dict)
        assert activations.shape == (batch, 2, neurons, position)

    def test_init(self):
        lm = ecco.from_pretrained('sshleifer/tiny-gpt2', activations=True)

        assert isinstance(lm.model, PreTrainedModel), "Model downloaded and LM was initialized successfully."

    def test_generate(self):
        lm = ecco.from_pretrained('sshleifer/tiny-gpt2',
                                  activations=True,
                                  verbose=False)
        output = lm.generate('test', generate=1, attribution=['grad_x_input'])

        assert output.token_ids.shape == (1, 2), "Generated one token successfully"
        assert output.attribution['grad_x_input'][0] == 1, "Successfully got an attribution value"
        # Confirm activations is dimensions:
        # (batch 1, layer 2, h_dimension 8, position 1)
        assert output.activations['decoder'].shape == (1, 2, 8, 1)

    def test_call_dummy_bert(self):
        lm = ecco.from_pretrained('julien-c/bert-xsmall-dummy',
                                  activations=True,
                                  verbose=False)
        inputs = lm.to(lm.tokenizer(['test', 'hi'],
                                    padding=True,
                                    truncation=True,
                                    return_tensors="pt",
                                    max_length=512))
        output = lm(inputs)
        # Confirm it's (batch 2, layer 1, h_dimension 40, position 3)
        # position is 3 because of [CLS] and [SEP]
        # If we do require padding, this CUDA compains with this model for some reason.
        assert output.activations['encoder'].shape == (2, 1, 40, 3)


    # TODO: Test LM Generate with Activation. Tweak to support batch dimension.
    # def test_generate_token_no_attribution(self, mocker):
    #     pass
    #
    # def test_generate_token_with_attribution(self, mocker):
    #     pass

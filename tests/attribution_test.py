from ecco.attribution import gradient_x_inputs_attribution
import torch
import pytest


@pytest.fixture
def simpleNNModel():
    class simpleNNModel(torch.nn.Module):
        def __init__(self):
            super(simpleNNModel, self).__init__()
            self.w = torch.tensor([[10., 10.]])

        def forward(self, x):
            return x * self.w
    yield simpleNNModel()


class TestAttribution:
    def test_grad_x_input(self, simpleNNModel):
        input = torch.tensor([[9., 9.]], requires_grad=True)
        output = simpleNNModel(input)
        expected = torch.tensor([1.])
        actual = gradient_x_inputs_attribution(output[0][0],input)
        assert torch.all(torch.eq(actual, expected))

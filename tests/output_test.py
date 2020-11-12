from ecco import output
import pytest

class TestOutput:
    def test_position_raises_value_error_more(self):
        outputSeq = output.OutputSeq(tokens=[0,0], n_input_tokens=1)

        with pytest.raises(ValueError):
            outputSeq.position(position=4)


    def test_position_raises_value_error_less(self):
        outputSeq = output.OutputSeq(tokens=[0,0], n_input_tokens=1)

        with pytest.raises(ValueError):
            outputSeq.position(position=0)

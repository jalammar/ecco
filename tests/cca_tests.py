from ecco import analysis
import pytest
import numpy as np

shape = (100, 1000)
np.random.seed(seed=1)


@pytest.fixture
def acts():
    acts1 = np.random.randn(*shape)
    acts2 = np.random.randn(*shape)
    yield acts1, acts2


class TestCCA:
    def test_cca_smoke(self, acts):
        actual = analysis.cca(acts[0], acts[1])
        assert isinstance(actual, float)
        assert actual >= 0
        assert actual <= 1

    def test_svcca_smoke(self, acts):
        actual = analysis.svcca(acts[0], acts[1])
        assert isinstance(actual, float)
        assert actual >= 0
        assert actual <= 1

    def test_pwcca_smoke(self, acts):
        actual = analysis.pwcca(acts[0], acts[1])
        assert isinstance(actual, float)
        assert actual >= 0
        assert actual <= 1

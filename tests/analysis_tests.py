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


class TestAnalysis:
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

    def test_cka_smoke(self, acts):
        actual = analysis.cka(acts[0], acts[1])
        assert isinstance(actual, float)
        assert actual >= 0
        assert actual <= 1

    def test_linear_transformation(self, acts):
        acts_1 = acts[0]
        acts_2 = acts_1 * 10
        assert pytest.approx(analysis.cca(acts_1, acts_2), 1.0), "CCA of linear transformation is approx 1.0"
        assert pytest.approx(analysis.svcca(acts_1, acts_2), 1.0), "SVCCA of linear transformation is approx 1.0"
        assert pytest.approx(analysis.pwcca(acts_1, acts_2), 1.0), "PWCCA of linear transformation is approx 1.0"
        assert pytest.approx(analysis.cka(acts_1, acts_2), 1.0), "CKA of linear transformation is approx 1.0"

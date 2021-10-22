import pytest
import numpy as np
from ecco import lm_plots
import os


class TestLMPlots:
    def test_save_rankings_plot(self, rankings_plot_data_1):
        lm_plots.plot_inner_token_rankings(**rankings_plot_data_1, save_file_path='./tmp/ranking_1.png')


    def test_save_ranking_watch_plot(self, ranking_watch_data_1):
        lm_plots.plot_inner_token_rankings_watch(**ranking_watch_data_1, save_file_path='./tmp/ranking_watch_1.png')


@pytest.fixture
def rankings_plot_data_1():
    yield {'input_tokens': ["'.'",
                            "' Denmark'",
                            "'\\n'",
                            "'5'",
                            "'.'",
                            "' Estonia'",
                            "'\\n'",
                            "'6'",
                            "'.'",
                            "' Hungary'"],
           'output_tokens': ["' Denmark'",
                             "'\\n'",
                             "'5'",
                             "'.'",
                             "' Estonia'",
                             "'\\n'",
                             "'6'",
                             "'.'",
                             "' Hungary'",
                             "'\\n'"],
           'rankings': np.array([[32716, 7, 162, 3, 35167, 10, 233, 3, 25890,
                                  7],
                                 [29856, 6, 169, 3, 32155, 6, 209, 3, 21194,
                                  4],
                                 [17906, 9, 163, 4, 22282, 9, 207, 4, 19788,
                                  7],
                                 [113, 1, 6, 1, 259, 1, 8, 1, 373,
                                  1],
                                 [14, 1, 1, 1, 29, 1, 1, 1, 60,
                                  1],
                                 [3, 1, 1, 1, 2, 1, 1, 1, 2,
                                  1]]),
           'predicted_tokens': np.array([[' Denmark', '\n', '5', '.', ' Estonia', '\n', '6', '.',
                                          ' Hungary', '\n'],
                                         [' Denmark', '\n', '5', '.', ' Estonia', '\n', '6', '.',
                                          ' Hungary', '\n'],
                                         [' Denmark', '\n', '5', '.', ' Estonia', '\n', '6', '.',
                                          ' Hungary', '\n'],
                                         [' Denmark', '\n', '5', '.', ' Estonia', '\n', '6', '.',
                                          ' Hungary', '\n'],
                                         [' Denmark', '\n', '5', '.', ' Estonia', '\n', '6', '.',
                                          ' Hungary', '\n'],
                                         [' Denmark', '\n', '5', '.', ' Estonia', '\n', '6', '.',
                                          ' Hungary', '\n']], dtype='<U25')}


@pytest.fixture
def ranking_watch_data_1():
    yield {'input_tokens': ['The', ' keys', ' to', ' the', ' cabinet', ';'],
           'output_tokens': ["' is'", "' are'"],
           'rankings': np.array([[19, 121],
                                 [15, 109],
                                 [16, 64],
                                 [14, 24],
                                 [9, 16],
                                 [12, 1]]),
           'position': 7}


@pytest.fixture(scope="session", autouse=True)
def tmp_dir():
    if not os.path.exists('tmp'):
        os.makedirs('tmp')

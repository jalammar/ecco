import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.cm import get_cmap
import copy
import warnings
from matplotlib.colors import ListedColormap
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
from typing import Optional, List, Dict
import sys


def plot_activations(tokens, activations, vmin=0, vmax=2, height=60,
                     width_scale_per_item=1,
                     file_prefix='neuron_activation_plot'):
    """ Plots a heatmap showing how active each neuron (row) was with each token
    (columns). Neurons with activation less then masking_threashold are masked.

    Args:
      tokens: list of the tokens. Note if you're examining activations
      associated with the token as input or as output.


    """
    n_tokens = activations.shape[-1]

    # Activations lower than this threshold will show up as blank
    masking_threshold = 0.01
    masked = np.ma.masked_less(activations, masking_threshold)
    # mask = tensor.shape()

    fig = plt.figure(figsize=(activations.shape[1], height))  # Width adjusts to number of tokens

    sns.set()
    ax = plt.gca()

    v = copy.copy(get_cmap('viridis_r'))
    v.set_bad('white')

    g = sns.heatmap(activations,
                    mask=masked.mask,
                    cmap=v,
                    ax=ax,
                    vmin=vmin,
                    vmax=vmax,
                    cbar=False)

    if tokens:
        ax.set_xticklabels(tokens[-n_tokens:], rotation=0)
    ax.tick_params(axis='x', which='major', labelsize=18)
    ax.set_xlabel('\nOutput Token', fontsize=14)

    plt.title('FFNN Activations', fontsize=28)
    plt.tick_params(axis='x',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    left=False,  # ticks along the bottom edge are off
                    bottom=True,  # ticks along the bottom edge are off
                    top=True,  # ticks along thx top edge are off
                    labeltop=True,
                    labelbottom=True)  # labels along the bottom edge are off

    plt.xticks(rotation=-45)

    # # Save Figure to file & download
    fig.set_facecolor("w")
    # time_str = int(datetime.datetime.now().strftime("%s")) * 1000
    # file_name = 'distilgpt2_layer_' + str(layer) + '_' + str(time_str) + '.png'
    # plt.savefig(file_name)
    # plt.close(fig)

    plt.show()


def plot_clustered_activations(tokens, activations, clusters, cluster_ids, file_prefix='neuron_activation_plot'):
    """ Plots a heat mapshowing how active each neuron (row) was with each token
    (columns). Neurons with activation less then masking_threashold are masked.

    Args:
      tokens: list of the tokens. Note if you're examining activations
      associated with the token as input or as output.


    """
    n_tokens = activations.shape[-1]
    # Activations lower than this threshold will show up as blank
    masking_threshold = 0.01
    masked = np.ma.masked_less(activations, masking_threshold)
    # mask = tensor.shape()

    fig = plt.figure(figsize=(activations.shape[1], 60))  # Width adjusts to number of tokens

    # sns.set()
    ax = plt.gca()

    v = copy.copy(get_cmap('viridis_r'))
    v.set_bad('white')
    g = sns.heatmap(activations,
                    mask=masked.mask,
                    cmap=v,
                    ax=ax,
                    vmax=2,
                    vmin=0,
                    cbar=False)

    colors = get_cmap("cool", len(clusters.keys()))
    colors = get_cmap("tab20", len(clusters.keys()))
    colors_2 = get_cmap("hot", len(clusters.keys()))
    colors_2 = get_cmap("prism", len(clusters.keys()))

    row = 0
    for idx, (cluster_id, neurons) in enumerate(clusters.items()):
        n_neurons = len(neurons)
        # print(idx, 'cluster #', cluster_id, "number of neurons: ",
        #       len(neurons), n_neurons, 'row', row)

        opacity = 0.00

        edge_color = colors(idx, 0.5)
        fill_color = edge_color  # colors(idx, opacity)

        # if idx % 2 == 0:
        #     edge_color = colors(idx)
        #     fill_color = colors(idx, opacity)  # Color + Opacity
        # else:
        #     edge_color = colors_2(idx)
        #     fill_color = colors_2(idx, opacity)  # Color + Opacity

        # First colored column to the leftmost of the figure identifying the cluster
        g.add_patch(Rectangle((-1, row),
                              1,  # width
                              n_neurons,  # height
                              fill=True,
                              facecolor=edge_color,
                              edgecolor=edge_color,
                              lw=5,
                              label="cluster {}".format(cluster_ids[idx])))

        # Border surrounding the cluster
        g.add_patch(Rectangle((-1, row),
                              activations.shape[1] + 1,  # width - span all columns
                              n_neurons,  # height
                              fill=False,
                              facecolor=fill_color,
                              edgecolor=edge_color,
                              lw=5))
        row += n_neurons

    # ax_ = g.ax_heatmap
    # print(ax_)

    # From https://github.com/mwaskom/seaborn/issues/1773
    # fix for mpl bug that cuts off top/bottom of seaborn viz
    left, right = plt.xlim()  # discover the values for bottom and top
    # right += 0.5  # Add 0.5 to the bottom
    left -= 0.5  # Subtract 0.5 from the top
    plt.xlim(left, right)  # update the ylim(bottom, top) values

    # print(ax_)

    if tokens:
        ax.set_xticklabels(tokens[-n_tokens:], rotation=0)
    ax.tick_params(axis='x', which='major', labelsize=28)

    plt.tick_params(axis='x',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    left=False,  # ticks along the bottom edge are off
                    bottom=True,  # ticks along the bottom edge are off
                    top=True,  # ticks along thx top edge are off
                    labeltop=True,
                    labelbottom=True)  # labels along the bottom edge are off

    plt.xticks(rotation=-45)

    # # Save Figure to file & download
    fig.set_facecolor("w")

    plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0))
    plt.show()


def token_barplot(tokens, values):
    fig, ax = plt.subplots(figsize=(len(values) / 2, 1))
    # fig = plt.figure(figsize=(len(values) / 2, 6))
    fig.set_facecolor("w")
    cm = get_cmap('viridis_r')
    colors = [cm(v / 0.5) for v in values]
    x = np.arange(len(values))
    bars = ax.bar(x, values, color=colors)
    # ax = sns.barplot(x=np.arange(len(values)), y=values, hue=values, palette='viridis')  # self.tokens[:len(importance)]
    ax.set_xticks(x)
    ax.set_xticklabels(tokens[:len(values)])
    ax.set_title('Feature importance when the model was generating the token: {}'
                 .format(tokens[len(values)]))  # repr(
    plt.xticks(rotation=-90)


# See: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
def plot_logit_lens(tokens,
                    softmax_scores,
                    predicted_tokens,
                    vmin=0,
                    vmax=1,
                    token_found_mask=None,
                    show_input_tokens: bool = False,
                    n_input_tokens: int = 0):
    # print(tokens, softmax_scores, predicted_tokens, token_found_mask)
    start_token = 0
    if not show_input_tokens:
        start_token = n_input_tokens - 1
        if n_input_tokens == 0:
            warnings.warn(
                'Setting show_input_tokens to True requires supplying n_input_tokens to exlucde inputs from the plot. Defaulting to 0 and showing the input.')

    fig = plt.figure(figsize=(20, 8))
    fig.set_facecolor("w")
    # Activations lower than this threshold will show up as blank
    # masking_threshold = 0.01
    # masked = np.ma.masked_less(softmax_scores, masking_threshold)
    ax = plt.gca()

    v = copy.copy(get_cmap('viridis_r'))
    v.set_bad('white')

    g = sns.heatmap(softmax_scores[:, start_token:],
                    mask=token_found_mask[:, start_token:],
                    cmap=v,
                    fmt="",
                    ax=ax,
                    annot=predicted_tokens[:, start_token:],
                    vmin=vmin,
                    vmax=vmax,
                    linewidths=0.5,
                    linecolor="#f0f0f0",
                    annot_kws={"size": 22},
                    cbar_kws={'label': 'Probability of token (softmax score)'}
                    )

    ax.tick_params(axis='x', which='major', labelsize=18)

    plt.tick_params(axis='x',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    left=False,  # ticks along the bottom edge are off
                    bottom=True,  # ticks along the bottom edge are off
                    top=False,  # ticks along thx top edge are off
                    labeltop=False,
                    labelbottom=True)  # labels along the bottom edge are off

    # Output token labels at the bottom
    ax.set_xticklabels(tokens[start_token + 1:], rotation=-90)

    # Layer names label at the left
    ylabels = ["Layer {}".format(n) for n in range(softmax_scores.shape[0])]
    ax.set_yticklabels(ylabels, fontsize=18, rotation=0)

    # Input token labels at the top
    ax2 = ax.twiny()
    ax2.set_xlim([0, ax.get_xlim()[1]])
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels(tokens[start_token:-1], fontsize=18, rotation=-90)


def plot_inner_token_rankings_watch(input_tokens,
                                    output_tokens,
                                    rankings: np.ndarray,
                                    position: int,
                                    vmin: Optional[int] = 2,  # Good range for topk 50
                                    vmax: Optional[int] = 5000,
                                    show_inputs: Optional[bool] = False,
                                    save_file_path: Optional[str] = None
                                    ):

    n_columns = len(output_tokens)
    n_rows = rankings.shape[0]
    fsize = (1 + 0.9 * n_columns,  # Make figure wider if more columns
             1 + 0.4 * n_rows)  # Make taller if more layers
    fig, (ax, cax) = plt.subplots(nrows=1, ncols=2,
                                  figsize=fsize,
                                  gridspec_kw={"width_ratios": [n_columns, 0.4]})
    fig.subplots_adjust(wspace=0.2)
    fig.set_facecolor("w")

    cmap_big = get_cmap('GnBu_r', 512)

    newcmp = ListedColormap(cmap_big(np.linspace(0.40, 0.90, 256)))
    v = copy.copy(newcmp)
    v.set_under('#1a7bb5')
    v.set_over('white')
    v.set_bad('white')

    comma_fmt = FuncFormatter(lambda x, p: format(int(x), ','))

    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    g = sns.heatmap(rankings,
                    # mask=token_found_mask[:,start_token:],
                    cmap=v,
                    fmt="d",
                    ax=ax,
                    annot=rankings,
                    cbar=False,
                    norm=norm,
                    linewidths=0.5,
                    linecolor="#f0f0f0",
                    annot_kws={"size": 12})

    fig.colorbar(ax.get_children()[0],
                 cax=cax,
                 format=comma_fmt,
                 extend='both',
                 orientation="vertical",
                 label='Ranking of token (by score)')

    ax.tick_params(axis='x', which='major', labelsize=20)
    plt.tick_params(axis='x',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    left=False,  # ticks along the bottom edge are off
                    bottom=True,  # ticks along the bottom edge are off
                    top=False,  # ticks along thx top edge are off
                    labeltop=False,
                    labelbottom=True)  # labels along the bottom edge are off

    # Output token labels at the bottom
    ax.set_xticklabels(output_tokens, rotation=-90)
    ax.set_xlabel('Output Token', fontsize=14)

    # Layer names label at the left
    ylabels = ["Decoder Layer {}".format(n) for n in range(rankings.shape[0])]
    ax.set_yticklabels(ylabels, fontsize=14, rotation=0)
    # ax.set_ylabel('Output Token')

    ax2 = ax.twiny()
    ax2.set_xlim([0, ax.get_xlim()[1]])
    if show_inputs:
        # Input token labels at the top
        ax2.set_xticks(ax.get_xticks())
        ax2.set_xticklabels(input_tokens, fontsize=14, rotation=-90)
        ax2.set_xlabel('\nWatched Token', fontsize=14)
    else:
        ax2.set_xticks([])

    plt.title(' '.join(input_tokens[:position + 1]) + ' ____\n', fontsize=14)

    if save_file_path is not None:
        try:
            plt.savefig(save_file_path)
        except:
            e = sys.exc_info()[0]
            print("<p>Error: (likely ./tmp/ folder does not exist or can't be created). %s</p>" % e)
            raise


def plot_inner_token_rankings(input_tokens,
                              output_tokens,
                              rankings,
                              vmin: int = 2,
                              vmax: int = 5000,
                              show_inputs: Optional[bool] = False,
                              save_file_path: Optional[str] = None,
                              **kwargs
                              ):

    n_columns = len(input_tokens)
    n_rows = rankings.shape[0]
    fsize = (1 + 0.9 * n_columns,  # Make figure wider if more columns
             1 + 0.4 * n_rows)  # Make taller if more layers

    fig, (ax, cax) = plt.subplots(nrows=1, ncols=2,
                                  figsize=fsize,
                                  gridspec_kw={"width_ratios": [n_columns, 0.5]}
                                  )
    plt.subplots_adjust(wspace=0.1)
    fig.set_facecolor("w")
    # ax = plt.gca()

    cmap_big = get_cmap('RdPu_r', 512)

    newcmp = ListedColormap(cmap_big(np.linspace(0.40, 0.90, 256)))
    v = copy.copy(newcmp)
    v.set_under('#9a017b')
    v.set_over('white')

    v.set_bad('white')

    comma_fmt = FuncFormatter(lambda x, p: format(int(x), ','))

    # print(vmin, vmax)
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    g = sns.heatmap(rankings,
                    # mask=token_found_mask[:,start_token:],
                    cmap=v,
                    fmt="d",
                    ax=ax,
                    annot=rankings,
                    cbar=False,
                    norm=norm,
                    linewidths=0.5,
                    linecolor="#f0f0f0",
                    annot_kws={"size": 12}
                    )

    fig.colorbar(ax.get_children()[0],
                 cax=cax,
                 format=comma_fmt,
                 extend='both',
                 orientation="vertical",
                 label='Ranking of token (by score)')

    ax.tick_params(axis='x', which='major', labelsize=18)

    plt.tick_params(axis='x',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    left=False,  # ticks along the bottom edge are off
                    bottom=True,  # ticks along the bottom edge are off
                    top=False,  # ticks along thx top edge are off
                    labeltop=False,
                    labelbottom=True)  # labels along the bottom edge are off

    # Output token labels at the bottom
    ax.set_xticklabels(output_tokens, rotation=-90)
    ax.set_xlabel('Output Token', fontsize=14)

    # Layer names label at the left
    ylabels = ["Layer {}".format(n) for n in range(rankings.shape[0])]
    ax.set_yticklabels(ylabels, fontsize=14, rotation=0)
    # ax.set_ylabel('Output Token')

    ax2 = ax.twiny()
    ax2.set_xlim([0, ax.get_xlim()[1]])
    if show_inputs:
        # Input token labels at the top
        ax2.set_xticks(ax.get_xticks())
        ax2.set_xticklabels(input_tokens, fontsize=14, rotation=-90)
        ax2.set_xlabel('\nInput Token', fontsize=14)
    else:
        ax2.set_xticks([])

    if save_file_path is not None:
        try:
            plt.savefig(save_file_path)
        except:
            e = sys.exc_info()[0]
            print("<p>Error: (likely ./tmp/ folder does not exist or can't be created). %s</p>" % e)
            raise

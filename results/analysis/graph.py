import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycountry


def mscatter(x, y, ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    ax = ax or plt.gca()
    sc = ax.scatter(x, y, **kw)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


color_map = {'all': 'red', 'stopwords': 'blue', 'content': 'green', 'function': 'black'}
offset_map = {'all': .35, 'stopwords': .116, 'content': -.116, 'function': -.35}
size_map = {'vanilla': 50, 'pos': 5, 'token': 5}
marker_map = {'vanilla': "o", 'pos': 7, 'token': 6}

df = pd.read_csv(
    'results/csv/oscar_size_50000000_n_shuffle_10.csv',
    names=[
        'language', 'shuffling', 'run', 'filter',
        'sparsity', 'Zipf R2', 'Zipf coeff'])

df['language_position'] = pd.factorize(df['language'])[0]
df['language_name'] = [pycountry.languages.get(alpha_2=lang_code).name
                       for lang_code in df['language']]


def scale_transformation(x):
    return -np.log(1-x)


df['Zipf R2 (transformed scale)'] = scale_transformation(df['Zipf R2'])

fig, ax = plt.subplots(1, 3, figsize=(12, 8))

for ix, metric in enumerate(['sparsity', 'Zipf R2 (transformed scale)', 'Zipf coeff']):
    mscatter(
            df[metric],
            df['language_position'] + [offset_map[f] for f in df['filter']],
            ax[ix],
            m=[marker_map[s] for s in df["shuffling"]],
            c=[color_map[f] for f in df['filter']],
            alpha=0.5,
            s=[size_map[s] for s in df["shuffling"]],
        )

    ax[ix].set_xlabel(metric)
    ax[ix].set_yticks([])
    ax[ix].set_yticklabels([])
    if metric == "sparsity":
        ax[ix].set_ylabel('language')
        ax[ix].set_yticks(df['language_position'].unique())
        ax[ix].set_yticklabels(df['language_name'].unique())
        ax[ix].set_xscale('log')
    if metric == "Zipf R2 (transformed scale)":
        x_values = np.array([.6, .7, .8, .9])
        ax[ix].set_xticks(scale_transformation(x_values))
        ax[ix].set_xticklabels(x_values)

    for i, language in enumerate(df['language'].unique()):
        if i != 0:
            ax[ix].hlines(i - 0.5,
                          xmin=df[metric].min(), xmax=df[metric].max(),
                          linestyle='dotted', linewidth=0.5, color='grey')

legend_handles = [
    mpatches.Patch(color=value, label=key)
    for key, value in color_map.items()]
fig.legend(
        title="Filtering:",
        handles=legend_handles, loc='upper right')

legend_markers = [
    plt.Line2D([0], [0], marker=value, color='w', label=key, markersize=10, markerfacecolor='black')
    for key, value in marker_map.items()]
fig.legend(
    title="Shuffling by:",
    handles=legend_markers, loc='center right')

plt.show()

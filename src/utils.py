
import tqdm
import pandas as pd
import numpy as np


def retrieve_text_from_wikipedia(wiki_data, idx):
    """
        Args:
            wiki_data [Hugging Face dataset] it contains several data, here we concatenate
                                             the texts from n_articles random articles.
            idx [np.array] array of articles to load.
        Returns:
            text [str]
    """
    # /!\ For now we load the whole text but might be improved in future versions /!\
    text = ''
    for i in idx:
        text += wiki_data[int(i)]['text']
    text = text.lower()
    return text

def compute_cooccurrence_matrix(bigram_counts, list_of_words, silent = False):
    """
        Compute co-occurrences matrix from bigram counts and the list of words.
        Rk: the list of words is not essential as it may be retrieve from the bigram counts
            but it fastens computation to feed it directly.
        Args:
            bigrams_count [dict] {('w1', 'w1'): N}
            list_of_words [List of str] ['w1',...,'wN']
        Returns:
            df [Pandas DataFrame] the cooccurrences matrix
    """
    coocurrence_dict = {}
    for w1 in tqdm.tqdm(list_of_words, total = len(list_of_words), disable = silent):
        vector = []
        for w2 in list_of_words:
            if w1 < w2:
                key = (w1, w2)
            else:
                key = (w2, w1)
            if key in bigram_counts.keys():
                vector.append(bigram_counts[key])
            else:
                vector.append(0)
        coocurrence_dict[w1] = vector
        
    df = pd.DataFrame(data = coocurrence_dict)
    df.index = list_of_words
    return df

def from_metrics2compute_to_metrics2plot(metrics2compute):
    """
        Kinda weird function that takes in argument the metrics to compute:
            ex: 
                   metrics2compute = ['sparsity', 'zipf_fit]
                => metrics2plot = {'sparsity': ['sparsity_prop_nonzeros'], 'zipf_fit': ['zipf_fit_R_sq', 'zipf_fit_coeff']}
        Rk: This is an hardcoded but necessary tool.
        Args:
            metrics2compute [List of str]
        Returns:
            metrics2plot [dict] 
    """

    metrics2plot = {}
    if 'sparsity' in metrics2compute:
        metrics2plot['sparsity'] = ['sparsity_prop_nonzeros']
    if 'zipf_fit' in metrics2compute:
        metrics2plot['zipf_fit'] = ['zipf_fit_R_sq',
                                    'zipf_fit_coeff']
    if 'clustering' in metrics2compute:
        metrics2plot['clustering'] = ['clustering_average',
                                      'clustering_transitivity']
        
    return metrics2plot

def convert_bigrams_to_ids(bigrams):
    """
        Take a list of bigrams and returns a list of bigrams ids.
            ex:
                [('a', 'b'), ('a','c'), ('b','d')] => [(0,1),(0,2),(1,3)]
    """
    token_to_id = {}
    id = 0
    new_list = []
    for w1, w2 in bigrams:
        if w1 in token_to_id.keys():
            id1 = token_to_id[w1]
        else:
            id1 = id
            token_to_id[w1] = id
            id += 1

        if w2 in token_to_id.keys():
            id2 = token_to_id[w2]
        else:
            id2 = id
            token_to_id[w2] = id
            id += 1
        new_list.append((id1, id2))
    return new_list


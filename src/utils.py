
import os
import tqdm
import pandas as pd
import numpy as np
import pickle

from datasets import load_dataset
import spacy

from config import spacy_lang_names


def retrieve_text_from_wikipedia(language: str,
                                 size: int,
                                 max_articles: int = 10e6):
    """
        Compute the texts from wikipedia according to idx.
        Returns also the idx of each articles in the final 
        text.
    
        Args:
            language [str]
            size [int]
            max_articles [int] buffer size from which we sample articles
        Returns:
            text [str]
            articles_idx [dict] keys: num [int]
                                values: (i, i+l)
                                i is the idx of the begining of the num^th article
                                and l is its length.
    """
    # Retrieve corpus
    if language == 'en':
        corpus = load_dataset("wikipedia", "20220301.en")['train']
    elif language == 'fr':
        corpus = load_dataset("wikipedia", "20220301.fr")['train']
    elif language == 'de':
        corpus = load_dataset("wikipedia", "20220301.de")['train']
    else:
        raise Exception(f"Language {language} is not supported for wikipedia.")
    
    # Compute idx
    idx = np.random.choice(len(corpus), max_articles)
    
    # /!\ For now we load the whole text but might be improved in future versions /!\
    text = ''
    articles_idx = {}
    prv_idx = 0 
    for num, i in enumerate(idx):
        _text = corpus[int(i)]['text']
        text += _text
        articles_idx[num] = (prv_idx, prv_idx + len(_text))
        prv_idx += len(_text) # prv_idx is equals to the current size of the text
        # Check for size limit
        if prv_idx >= size:
            break
    text = text.lower()
    return prv_idx, text, articles_idx

def retrieve_text_from_oscar(language: str,
                             size: int,
                             max_articles: int = 20000):
    """
        Compute the texts from OSCAR according to idx.
        Returns also the idx of each articles in the final 
        text.
    
        Args:
            language [str]
            size [int]
            max_articles [int] buffer size from which we sample articles
        Returns:
            text [str]
            articles_idx [dict] keys: num [int]
                                values: (i, i+l)
                                i is the idx of the begining of the num^th article
                                and l is its length.
    """
    # Retrieve corpus
    unshuffled_corpus = load_dataset("oscar-corpus/OSCAR-2201",
                          use_auth_token="***REMOVED***", # required
                          language=language, 
                          streaming=True, # optional
                          split="train") 
    corpus = unshuffled_corpus.shuffle(buffer_size=max_articles)#, seed=42)
                                                              # No seed, also buffer_size big
    
    # /!\ For now we load the whole text but might be improved in future versions /!\
    text = ''
    articles_idx = {}
    prv_idx = 0 
    for num, d in enumerate(corpus):
        #print(f"Num: {num}; Size: {prv_idx}")
        _text = d['text']
        text += _text
        articles_idx[num] = (prv_idx, prv_idx + len(_text))
        prv_idx = prv_idx + len(_text)
        # Check for size limit
        if prv_idx >= size:
            break
    text = text.lower()
    return prv_idx, text, articles_idx

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

def load_spacy(language: str):
    """
    Load spacy according to the specified language.
    Download is not present.

    Args:
        language [str] 
    Returns:
        nlp [spacy thing]
    """
    
    spacy_name = spacy_lang_names[language]
   
    try: 
        nlp=spacy.load(spacy_name)
    except:
        print(f"Installing {spacy_name}...")
        os.system(f"python -m spacy download {spacy_name}")
        nlp=spacy.load(spacy_name)
        
    nlp.max_length = 100000000
    
    return nlp

def get_save_folder_name(
                  language: str,
                  size: int,
                  n_shuffle: int,
                  dataset: str,
                  split: str,
                  ordered: bool,
                  window_size: int):

    # Boring stuff
    filename = ""
    
    if language is None:
        # Add params to the name
        if split == 'sentence':
            filename += f'{dataset}_size_{int(size)}_n_shuffle_{n_shuffle}_sentence'
        elif split == 'window':
            filename += f'{dataset}_size_{size}_n_shuffle_{n_shuffle}_window_{window_size}'
        else:
            raise Exception(f"Split {split} doesn't exist.")
    else:
        # Add params to the name
        if split == 'sentence':
            filename += f'{dataset}_{language}_size_{size}_n_shuffle_{n_shuffle}_sentence'
        elif split == 'window':
            filename += f'{dataset}_{language}_size_{size}_n_shuffle_{n_shuffle}_window_{window_size}'
        else:
            raise Exception(f"Split {split} doesn't exist.")
        
    if ordered:
        filename += "_ordered"
    
    return filename


def save_analysis(analysis: dict,
                  language: str,
                  size: int,
                  n_shuffle: int,
                  dataset: str,
                  split: str,
                  ordered: bool,
                  window_size: int):
    """
        Save the analysis dict in results\\dicts, one dict per language.
        Args:
            analysis [dict] analysis dict
            languages [str] language to save
            size [int] number of articles read
            n_shuffle [int] number of random points for bootstrapping
            dataset [str] name of the dataset used
    
    """
    print("Saving analysis...")
    
    filename = os.path.join("results", "dicts")
    os.makedirs(filename, exist_ok=True)
    filename = os.path.join(filename, "")

    filename += get_save_folder_name(language, size, n_shuffle, dataset, split, ordered, window_size)

    # Save    
    with open('%s.pkl'%filename, 'wb') as file:
        pickle.dump(analysis[language], file)

    print("Done!")

def from_dict_to_csv(res_dict: dict = None,
                     languages: list = None,
                     size: int = None,
                     n_shuffle: int = None,
                     dataset: str = None,
                     filters: list = None,
                     split: str = 'sentence',
                     ordered: bool = False,
                     window_size: int = 5):
    """
    Convert analysis dict to friendly reading csv. If no dict is provided 
    it assumes it has already been saved to pickle and load it from there.

    Args:
        res_dict (dict, optional): analysis dict. {'en': {'vanilla': {'sparsity_prop_nonzeros': 0.1,
                                                                      'zipf_fit_coeff': -0.6,
                                                                      'zipf_fit_R_sq': 0.9,
                                                                      'filtered_stopwords_sparsity_prop_nonzeros': 0.1,
                                                                      'filtered_stopwords_zipf_fit_coeff': -0.6,
                                                                      'filtered_stopwords_zipf_fit_R_sq': 0.9},
                                                            'token_shuffled_0': {'sparsity_prop_nonzeros': 0.1,
                                                                                 'zipf_fit_coeff': -0.6,
                                                                                 'zipf_fit_R_sq': 0.9,
                                                                                 'filtered_stopwords_sparsity_prop_nonzeros': 0.1,
                                                                                 'filtered_stopwords_zipf_fit_coeff': -0.6,
                                                                                 'filtered_stopwords_zipf_fit_R_sq': 0.9},
                                                            'pos_shuffled_0': {'sparsity_prop_nonzeros': 0.1,
                                                                               'zipf_fit_coeff': -0.6,
                                                                               'zipf_fit_R_sq': 0.9,
                                                                               'filtered_stopwords_sparsity_prop_nonzeros': 0.1,
                                                                               'filtered_stopwords_zipf_fit_coeff': -0.6,
                                                                               'filtered_stopwords_zipf_fit_R_sq': 0.9}
                                                           }
                                                    }
        languages (list of str, optional)
        size (int, optional)
        n_shuffle (int, optional)
        dataset [str]
        filters [list] not optional
    """
    
    assert (res_dict is not None) or (languages is not None)
    assert filters is not None
    
    print("Converting Results in .csv")
    
    # If no res_dict is provided then load it from pickle
    if ordered:
        ordered_str = "_ordered"
    else:
        ""
    
    if res_dict is None:
        if split == 'sentence':
            filename = os.path.join("results", 
                                    "dicts", 
                                    f"{dataset}_lang_size_{size}_n_shuffle_{n_shuffle}_sentence{ordered_str}")
        elif split == 'window':
            filename = os.path.join("results", 
                                    "dicts", 
                                    f"{dataset}_lang_size_{size}_n_shuffle_{n_shuffle}_window_{window_size}{ordered_str}")
        res_dict = {}
        for lang in languages:
            with open('%s.pkl'%filename.replace('lang', lang), 'rb') as file:
                _res = pickle.load(file)
            res_dict[lang] = _res
            
    ## Now let's get down to it ##
    
    if split == 'sentence':
        csv_name = os.path.join("results", 
                                "csv", 
                                f"{dataset}_size_{size}_n_shuffle_{n_shuffle}_sentence{ordered_str}.csv")
    if split == 'window':
        csv_name = os.path.join("results", 
                                "csv", 
                                f"{dataset}_size_{size}_n_shuffle_{n_shuffle}_window_{window_size}{ordered_str}.csv")
    
    # First we create data
    # create columns
    columns = ['lang', # en, fr, ... 
                'split', # vanilla, token, pos
                'num', # num of the split
                'filter', # all, stopwords, content, function
                'sparsity', # sparsity_prop_nonzeros
                'zipf fit', # zipf_fit_R_sq
                'zipf coeff', # zipf_fit_coeff
                ] 
    data = {col: [] for col in columns}
    for lang in res_dict.keys():
        for sp in res_dict[lang].keys():
            if sp == 'vanilla':
                split_name = 'vanilla'
                num = 0
            elif 'token_shuffled' in sp:
                split_name = 'token'
                num = int(sp[15:])
            elif "pos_shuffled" in sp:
                split_name = 'pos'
                num = int(sp[13:])
            else:
                continue
                
            # add all data
            data['lang'].append(lang)
            data['split'].append(split_name)
            data['num'].append(num)
            data['filter'].append('all')
            data['sparsity'].append(res_dict[lang][sp]['sparsity_prop_nonzeros'])
            data['zipf fit'].append(res_dict[lang][sp]['zipf_fit_R_sq'])
            data['zipf coeff'].append(res_dict[lang][sp]['zipf_fit_coeff'])
            
            for filter in filters:
                data['lang'].append(lang)
                data['split'].append(split_name)
                data['num'].append(num)
                data['filter'].append(filter)
                data['sparsity'].append(res_dict[lang][sp]['filtered_' + filter + '_sparsity_prop_nonzeros'])
                data['zipf fit'].append(res_dict[lang][sp]['filtered_' + filter + '_zipf_fit_R_sq'])
                data['zipf coeff'].append(res_dict[lang][sp]['filtered_' + filter + '_zipf_fit_coeff'])
                
    df = pd.DataFrame(data)
            
    # First we need to know if we create it from scratch or no
    if not(os.path.exists(csv_name)):
        try:
            # create csv dir
            os.mkdir(os.path.join("results",
                                  "csv"))
        except:
            pass 
            # the folder already exists
        
        df.to_csv(csv_name, index=False, header=False)
    else:  
        # otherwise append 
        df.to_csv(csv_name, mode='a', index=False, header=False)

    print("Done!")

import tqdm
import itertools
from collections import Counter
import pandas as pd
import numpy as np
import string
from typing import Dict, Tuple, List, Union

from src.utils import load_spacy

class Cooccurrences:
    """
        Wrapper class for co-occurrences computation.
    """

    def __init__(self, language: str, 
                       split: str = 'sentence',
                       ordered: bool = False, 
                       window_size: int = 5,
                       silent: bool = False):
        # Data
        self.text = None
        
        # Split
        self.split = split
        self.window_size = window_size
        self.ordered = ordered
        
        # Form
        self.silent = silent
        
        # Tentative with SpaCy
        self.language = language
        self.nlp = load_spacy(language = language) # Candidate for word tokenization
        # We remove everything in the pipeline
        # that is not useful, here everything
        for name, _ in self.nlp.pipeline:
            self.nlp.remove_pipe(name)
        self.nlp.add_pipe("sentencizer") # We need sentence tokenization

        # Usefull to store
        self.list_of_words = None

    def update_text(self, text, just_shuffled=True):
        """
            Update the text attribute from the class.
            If just_shuffled = True it means that the new text is just a shuffled
            version of the old one.
            Args:
                text [str]
                just_shuffled [bool]
        """
        self.text = text
        if not(just_shuffled):
            # for a shuffled text the list of words doesn't change...
            self.list_of_words = None

    def compute_feed_dict(self) -> Dict[str, Union[List[str], Dict[str, int]]]:
        """
            Main method of the class.
            Compute different quantities that will be usefull for metrics computation.
            
            Returns:
                feed_dict [dict]
        """
        feed_dict = {}

        print("\t\tComputing list of words and bigrams count...")
        
        list_of_words, bigrams_count = self._compute_list_of_words_and_bigrams()
        #feed_dict['bigrams_count'] = self._count_bigrams(list_of_bigrams)
        
        feed_dict['bigrams_count'] = bigrams_count
        feed_dict['list_of_words'] = list_of_words

        return feed_dict

    
    def _compute_bigrams(self, list_of_words) -> List[Tuple[str, str]]:
        """
            Compute bigrams from a tokenized sentence called here list_of_words.
            Rk: It has nothing to do with the real list_of_words computed by the
                method compute_list_of_words. 
            Args:
                list_of_words [List of str] ex: ['a', 'c', 'b']
            Returns:
                list_of_bigrams [List of tuple] If self.ordered: [('a','c'),  ('a', 'b'), ('c', 'b')]
                                                Else: [('a','c'), ('a', 'b'), ('b', 'c')]
        """
        list_of_bigrams = []
        for k in range(len(list_of_words)):
            w1 = list_of_words[k]
            if w1 in string.punctuation or w1 in ['\n', '\t']:
                continue
            for l in range(k+1, len(list_of_words)):
                w2 = list_of_words[l]
                if w2 in string.punctuation or w2 in ['\n', '\t']:
                    continue
                if self.ordered:
                    list_of_bigrams.append((w1, w2)) # Here whatever the alphabetic order we store the bigrams as it was in the text
                else:
                    if w1 < w2: # every bigram is in alphabetic order this way ('a', 'b') and ('b', 'a') count for the same bigram
                        list_of_bigrams.append((w1, w2))
                    else:
                        list_of_bigrams.append((w2, w1))
        return list_of_bigrams

    def _compute_list_of_words_and_bigrams(self, 
                                           n_subsets = 30, 
                                           max_sentence_size = 100) -> Tuple[List[str], Dict[Tuple[str], int]]:
        """
            Returns list of all bigrams.
            
            Args:
                n_subsets [int] divide text in subtexts to process it (memory issues)
                max_sentence_size [int] because of chinese!

            Returns:
                list_of_words [List of str] ex: ['a', 'b', 'c']
                list_of_bigrams [List of List of bigrams] ex: [[('a', 'b'), ('a', 'a'), ('a', 'b')],
                                                               [('a', 'c')]]
        """
        bigrams_count = Counter()
        set_of_words = set()
        
        ####
        mean_sentence_length = 0
        num_sentences = 0
        
        ### Not clean but f it I don't have time nor envy to do better
        
        # We can't load the full text in SpaCy so let's split it randomly!
        n_car = len(self.text)
        subsets = np.arange(0, 
                             n_car, 
                             n_car//n_subsets)
        
        for k, i1 in enumerate(subsets):
            
            try:
                i2 = subsets[k+1]
            except:
                i2 = n_car
            _text = self.text[i1:i2]
        
            list_of_words = [] # buffer that contains sentences
            
            for tok in self.nlp(_text):
                # For list_of_words
                set_of_words.add(tok.text)
                
                # add to list of words
                list_of_words.append(tok.text)
                
                if self.split == 'sentence':
                    if tok.is_sent_end or len(list_of_words) >= max_sentence_size: # the second condition is to avoid memory issues
                        ###
                        mean_sentence_length += len(list_of_words)
                        num_sentences += 1
                        
                        # Here we compute cooc
                        bigrams_count.update(
                            self._compute_bigrams(
                                list_of_words #self.word_tokenizer(line)
                                )
                            )
                        # reset sentence buffer
                        list_of_words = []
                elif self.split == 'window':
                    if len(list_of_words) >= self.window_size:
                        # Here we compute cooc
                        bigrams_count.update(
                            self._compute_bigrams(
                                list_of_words #self.word_tokenizer(line)
                                )
                            )
                        # Here, as it is a sliding window, we do not reset list of words
                        # We pop the first elem out that's all 
                        list_of_words = list_of_words[1:]
                else:
                    raise Exception(f"Split {self.split} is not defined.")
                
        print(f"Num words: {len(set_of_words)}")
        print(f"Num sentences: {num_sentences}")
        print(f"Avg. length: {mean_sentence_length/num_sentences}")
        
            
        return list(set_of_words), bigrams_count


    def _count_bigrams(self, list_of_bigrams):
        """
        
            /!\ Outdated /!\ 
            
            Compute the bigram counts from a list of bigrams.
            Args:
                list_of_bigrams [List of List of bigrams]
            Returns:
                bigram_counts [collections.Counter] Counter({('a','b'): 5, ('a', 'a'): 2})
        """
        # Flatten list of bigrams
        bigrams = list(itertools.chain(*list_of_bigrams))
        # Create counter of words in clean bigrams
        bigram_counts = Counter(bigrams) # counter: {(w1, w2): count}
        return bigram_counts
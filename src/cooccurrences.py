
import tqdm
import itertools
import collections
import pandas as pd
import numpy as np

from src.utils import load_spacy

class Cooccurrences:
    """
        Wrapper class for co-occurrences computation.
    """

    def __init__(self, language: str, silent: bool = False):
        # Data
        self.text = None
        
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

    def compute_feed_dict(self):
        """
            Main method of the class.
            Compute different quantities that will be usefull for metrics computation.
            
            Returns:
                feed_dict [dict]
        """
        feed_dict = {}

        print("\t\tComputing list of words and bigrams count...")
        
        list_of_words, list_of_bigrams = self._compute_list_of_words_and_bigrams()
        feed_dict['bigrams_count'] = self._count_bigrams(list_of_bigrams)
        
        feed_dict['list_of_words'] = list_of_words

        return feed_dict

    
    def _compute_bigrams(self, list_of_words):
        """
            Compute bigrams from a tokenized sentence called here list_of_words.
            Rk: It has nothing to do with the real list_of_words computed by the
                method compute_list_of_words. 
            Args:
                list_of_words [List of str] ex: ['a', 'b', 'c']
            Returns:
                list_of_bigrams [List of tuple] ex: [('a','b'),  ('a', 'c'), ('b', 'c')]
        """
        list_of_bigrams = []
        for k in range(len(list_of_words)):
            w1 = list_of_words[k]
            for l in range(k+1, len(list_of_words)):
                w2 = list_of_words[l]
                if w1 < w2: # every bigram is in alphabetic order this way ('a', 'b') and ('b', 'a') count for the same bigram
                    list_of_bigrams.append((w1, w2))
                else:
                    list_of_bigrams.append((w2, w1))
        return list_of_bigrams

    def _compute_list_of_words_and_bigrams(self, n_subsets = 30):
        """
            Returns list of all bigrams.

            Returns:
                list_of_words [List of str] ex: ['a', 'b', 'c']
                list_of_bigrams [List of List of bigrams] ex: [[('a', 'b'), ('a', 'a'), ('a', 'b')],
                                                               [('a', 'c')]]
        """
        list_of_bigrams = []
        set_of_words = set()
        
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
                
                if tok.is_sent_end:
                    # Here we compute cooc
                    list_of_bigrams.append(
                        self._compute_bigrams(
                            list_of_words #self.word_tokenizer(line)
                            )
                        )
                    # reset sentence buffer
                    list_of_words = []
            
        return list(set_of_words), list_of_bigrams


    def _count_bigrams(self, list_of_bigrams):
        """
            Compute the bigram counts from a list of bigrams.
            Args:
                list_of_bigrams [List of List of bigrams]
            Returns:
                bigram_counts [collections.Counter] Counter({('a','b'): 5, ('a', 'a'): 2})
        """
        # Flatten list of bigrams in clean tweets
        bigrams = list(itertools.chain(*list_of_bigrams))
        # Create counter of words in clean bigrams
        bigram_counts = collections.Counter(bigrams) # counter: {(w1, w2): count}
        return bigram_counts

import tqdm
import itertools
import collections
import pandas as pd
import numpy as np

from nltk.tokenize import RegexpTokenizer, sent_tokenize


class Cooccurrences:
    """
        Wrapper class for co-occurrences computation.
    """

    def __init__(self, silent = False):
        # Data
        self.text = None
        
        # Form
        self.silent = silent
        
        # Tokenizers
        self.word_tokenizer = RegexpTokenizer(r'\w+').tokenize
        self.sentence_tokenizer = sent_tokenize

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

    def compute_feed_dict(self,
                          bigram_counts=True,
                          list_of_words=True):
        """
            Main method of the class.
            Compute different quantities that will be usefull for metrics computation.
            Args:
                bigram_counts [bool]
                list_of_words [bool]
            Returns:
                feed_dict [dict]
        """
        feed_dict = {}

        if bigram_counts:
            print("\t\tComputing bigrams count...")
            feed_dict['bigrams_count'] = self._compute_bigrams_count()
        if list_of_words:
            if self.list_of_words: # Already been computed
                feed_dict['list_of_words'] = self.list_of_words
            print("\t\tComputing list of words...")
            self.list_of_words = self._compute_list_of_words()
            feed_dict['list_of_words'] = self.list_of_words

        return feed_dict

    def _compute_bigrams_count(self):
        """
            From an untokenized text,compute its bigrams count.
            Returns:
                bigrams_count [collections.Counter] Counter({('a','b'): 5, ('a', 'a'): 2})
        """

        list_of_bigrams = self._compute_list_of_bigrams(
                            list_of_lines = self.sentence_tokenizer(self.text),
                            silent = self.silent
                            )
        return self._count_bigrams(list_of_bigrams)

    def _compute_list_of_words(self):
        """
            From an untokenized text,compute the list of all its words.
            Rk: if a word appears more than one time in the text it will
                still appear only once in the list.
            Returns:
                list_of_words [List of str] ex: ['w1',...,'wN'] with i=/=j => wi =/= wj
        """
        tokenized_text = self.word_tokenizer(
                                    self.text.lower()
                                    )
        return list(set(tokenized_text))
    
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

    def _compute_list_of_bigrams(self, list_of_lines, silent = False):
        """
            Returns list of all bigrams.
            Args:
                list_of_lines [List of str] ex: ['a b a', 'a c']
            Returns:
                list_of_bigrams [List of List of bigrams] ex: [[('a', 'b'), ('a', 'a'), ('a', 'b')],
                                                               [('a', 'c')]]
        """
        list_of_bigrams = []
        for line in tqdm.tqdm(list_of_lines, disable = silent):
            list_of_bigrams.append(
                self._compute_bigrams(
                    self.word_tokenizer(line)
                    )
                )
        return list_of_bigrams


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
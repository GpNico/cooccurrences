
import numpy as np
import tqdm
import random

from nltk.tokenize import word_tokenize

from src.utils import load_spacy


class Shuffler:
    """
        Wrapper Class for the shuffles operations.
    """

    def __init__(self, language):
        self.pos_sequence = None
        self.E_sets = None
        
        # Language
        self.language = language

        # NLP from SpaCy
        self.nlp = load_spacy(language)
        self.nlp.remove_pipe("ner") # it seems to be the only useless thing

    def token_shuffle(self, text):
        """
            Shuffle the corpus at token level.
        """
        tokenized_text = word_tokenize(text)
        random.shuffle(tokenized_text)
        return ' '.join(tokenized_text)
    
    def pos_shuffle(self, text):
        """
            Shuffle the corpus at with respect to its POS structure.
        """
        if self.pos_sequence:
            assert self.E_sets
        else:
            print("\t\tCompute POS sequence and Es sets...")
            self.compute_pos_sequence_and_E_sets(text)

        self._shuffle_E_sets()
        return self._generate_text_from_pos()
    
    def compute_pos_sequence_and_E_sets(self, text: str,
                                              articles_idx: dict,
                                              n_subsets: int = 10):
        """"
            For a text [str] 'a b c' we note [p_a, p_b, p_c] its POS
            sequence. E(a) = {w|POS(w) = a} so here E_a = [a] if p_a =/= p_b, p_c.
            Args:
                text [str]
                articles_idx [dict] keys: num [int]
                                    values: (i, i+l)
                                    i is the idx of the begining of the num^th article
                                    and l is its length.
                n_subtexts [int] number in which we divide text
        """
        self.pos_sequence = []
        self.E_sets = {}
        # Here we can't do this directly because of Segmentation Fault
        # we need to devide text in subtexts.
        n_articles = len(articles_idx)
        
        articles = np.arange(0, 
                             n_articles, 
                             max(1, n_articles//n_subsets))

        for k in tqdm.tqdm(range(len(articles))):
            article = articles[k]
            i1, _ = articles_idx[article]
            try:
                next_article = articles[k+1]
                i2, _ = articles_idx[next_article]
            except:
                # We reach the end of the loop
                next_article = n_articles - 1
                _, i2 = articles_idx[next_article]

            _text = text[i1:i2]
            for token in self.nlp(_text):
                # Compute detailled POS and Dependency Tag
                tags = '{}-{}'.format(token.tag_, token.dep_)
                self.pos_sequence.append(tags)
                
                if tags in self.E_sets.keys():
                    self.E_sets[tags].append(token.text)
                else:
                    self.E_sets[tags] = [token.text]

    def _shuffle_E_sets(self):
        """
            In order to shuffle the dataset we shuffle the E_s sets.
        """  
        for k in self.E_sets.keys():
            random.shuffle(self.E_sets[k])

    def _generate_text_from_pos(self):
        """
            Take a POS sequence as well as Es sets and returns
            a text that respect the POS sequence.
            Rk: the Es sets are used to replace the POS tags in
                the order of their list. Hence the necessity to 
                shuffle them.
        """
        idx_Es = {k: 0 for k in self.E_sets.keys()} # To keep track where we are
        text = ''
        for pos in self.pos_sequence:
            new_token = self.E_sets[pos][idx_Es[pos]]
            idx_Es[pos] += 1
            text += new_token + ' '
        return text
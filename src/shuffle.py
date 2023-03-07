
from nltk.tokenize import word_tokenize
import spacy

import random



class Shuffler:
    """
        Wrapper Class for the shuffles operations.
    """

    def __init__(self, language):
        self.pos_sequence = None
        self.E_sets = None

        # NLP from SpaCy
        if language == 'en':
            self.nlp=spacy.load('en_core_web_sm')
        elif language == 'fr':
            self.nlp=spacy.load('fr_core_news_sm')
        elif language == 'de':
            self.nlp=spacy.load('de_core_news_sm')
        self.nlp.max_length = 4000000

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
            self._compute_pos_sequence_and_E_sets(text)

        self._shuffle_E_sets()
        return self._generate_text_from_pos()
    
    def _compute_pos_sequence_and_E_sets(self, text):
        """"
            For a text [str] 'a b c' we note [p_a, p_b, p_c] its POS
            sequence. E(a) = {w|POS(w) = a} so here E_a = [a] if p_a =/= p_b, p_c.
            Args:
                text [str]
        """
        self.pos_sequence = []
        self.E_sets = {}
        for token in self.nlp(text):
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
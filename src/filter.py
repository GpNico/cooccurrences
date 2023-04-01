
from collections import Counter

import numpy as np
from nltk.corpus import stopwords
import spacy

class Filter:
    """
        Class used to filter bigrams and list of words according to
        a POS list or nltk stopwords before going through the Metrics
        object.
    """

    def __init__(self, language: str,
                       filters: list = []):
        
        self.filters = filters
        self.language = language

        if language == 'en':
            self.stopwords = stopwords.words('english')
            self.nlp=spacy.load('en_core_web_sm')
        elif language == 'fr':
            self.stopwords = stopwords.words('french')
            self.nlp=spacy.load('fr_core_news_sm')
        elif language == 'de':
            self.stopwords = stopwords.words('german')
            self.nlp=spacy.load('de_core_news_sm')
        else:
            raise Exception("This language %s is not supported." % language)
            
        self._load_content_logical_pos_lists()
        # This only have to be computd once!
        self.words_pos = None
        
    def filter(self, feed_dict, replace = False):
        """
            Main method. Filter the feed_dict by creating
            two new quantities: 
                - filtered_bigrams_count
                - filterd_list_of_words
            where it respectively removed bigrams containing 
            stopwords and removing stopwords.
            If replace == True then the original feed_dict
            is replaced by the filtered one. Otherwise it just
            adds keys.
            Args:
                feed_dict [dict] containing at least bigrams_count and
                                 list_of_words 
        """

        print("\t\tFiltering...")

        assert 'bigrams_count' in feed_dict.keys()
        assert 'list_of_words' in feed_dict.keys()


        # Only have to compute words_pos once per language
        if self.words_pos:
            pass
        else:
            self._process_list_of_words(list_of_words = feed_dict['list_of_words'])

        ### Beginning list_of_words filtering

        filtered_list_of_words = {filter: [] for filter in self.filters}
        num_filtered = {filter: 0 for filter in self.filters}

        for w in feed_dict['list_of_words']:

            # due to tokenization issues some words may be absent
            # autmatically filtering them out
            try:
                pos = self.words_pos[w]
            except:
                for filter in self.filters:
                    num_filtered[filter] += 1
                continue

            ## Filter what is asked
            
            # stopwords
            if w in self.stopwords:
                num_filtered['stopwords'] += 1
            else:
                filtered_list_of_words['stopwords'].append(w)

            # content
            if not(pos in self.content_pos_list):
                num_filtered['content'] += 1
            else:
                filtered_list_of_words['content'].append(w)
            
            # function
            if not(pos in self.function_pos_list):
                num_filtered['function'] += 1
            else:
                filtered_list_of_words['function'].append(w)
        
        # Add filtered data to feed_dict

        for filter in self.filters:
            
            # print filtered info
            prop_filtered = np.round(100*num_filtered[filter]/len(feed_dict['list_of_words']), 2)
            print(f"\t\t\t{filter} filtered out {num_filtered[filter]} words ( {prop_filtered} % ).")
            
            # add
            feed_dict[f'filtered_{filter}_list_of_words'] = filtered_list_of_words[filter]
        
        
        ### Beginning of bigrams_count_filtering
        
        filtered_bigrams_count = {filter: Counter() for filter in self.filters}
        num_filtered = {filter: 0 for filter in self.filters}
        
        for w1, w2 in feed_dict['bigrams_count'].keys():
            
            # due to tokenization issues some words may be absent
            # autmatically filtering them out
            try:
                pos1, pos2 = self.words_pos[w1], self.words_pos[w2]
            except:
                for filter in self.filters:
                    num_filtered[filter] += 1
                continue

            count = feed_dict['bigrams_count'][(w1, w2)]
            
            ## Filter what is asked
            
            # stopwords
            if w1 in self.stopwords or w2 in self.stopwords:
                num_filtered['stopwords'] += 1
            else:
                filtered_bigrams_count['stopwords'][(w1,w2)] = count

            # content
            if not(pos1 in self.content_pos_list) or not(pos2 in self.content_pos_list):
                num_filtered['content'] += 1
            else:
                filtered_bigrams_count['content'][(w1,w2)] = count

            # function
            if not(pos1 in self.function_pos_list) or not(pos2 in self.function_pos_list):
                num_filtered['function'] += 1
            else:
                filtered_bigrams_count['function'][(w1,w2)] = count

        # Add filtered data to feed_dict

        for filter in self.filters:
            
            # print filtered info
            prop_filtered = np.round(100*num_filtered[filter]/len(feed_dict['bigrams_count']), 2)
            print(f"\t\t\t{filter} filtered out {num_filtered[filter]} bigrams ( {prop_filtered} % ).")
            
            # add
            feed_dict[f'filtered_{filter}_bigrams_count'] = filtered_bigrams_count[filter]

        
    def _filter_function(self, feed_dict):
        """
            Filter function by keeping only function words.

            /!\ The behavior is thus opposed to _filter_stopwords that
                keep only non-stopwords /!\
        """
        pass

    def _filter_content_function(self, feed_dict):
        """
            
        """
        pass

    def _load_content_logical_pos_lists(self):
        """
            Create lists of content & logical detailled
            POS (.tag_) from SpaCy.
        """
        if self.language == 'en':
            self.content_pos_list = ['AFX', 
                                    'JJ', 
                                    'JJR',
                                    'JJS', 
                                    'NN', 
                                    'NNS', 
                                    'WP',
                                    'VB',
                                    'VBD',
                                    'VBG',
                                    'VBN',
                                    'VBP',
                                    'VBZ']
            
            self.function_pos_list = ['IN',
                                      'EX',
                                      'CC',
                                      'DT',
                                      'TO',
                                      'PRP',
                                      'BES',
                                      'HVS',
                                      'MD']
            

        else:
            # It seems like spacy doesn't support detailled
            # POS for french
            self.content_pos_list = ['NOUN', 'ADJ', 'VERB']

            self.function_pos_list = ['CONJ', 'DET', 'ADP']

    def _process_list_of_words(self, list_of_words):
        """
            Compute detailled POS tag from .
            Args:
                list_of_words [list of str] ex: ['pizza', 'dogs', 'be', 'far']
            Returns:
                words_pos [dict] keys: words
                                 value: POS
                                 ex: {'pizza': NN,
                                      'dogs': NNS,
                                      'be': VB,
                                      'far': RN}
        """

        print("\t\t\tCompute list_of_words POS...")
        
        self.words_pos = {}
        processed = self.nlp(' '.join(list_of_words))
        for tok in processed:
            if self.language == 'en':
                self.words_pos[tok.text] = tok.tag_
            elif self.language in ['fr', 'de']:
                self.words_pos[tok.text] = tok.pos_
        
        print("\t\t\tDone!")


from collections import Counter

import numpy as np
from nltk.corpus import stopwords

class Filter:
    """
        Class used to filter bigrams and list of words according to
        a POS list or nltk stopwords before going through the Metrics
        object.
    """

    def __init__(self, language,
                       nltk_stopwords = True,
                       pos_list = None):
        self.nltk_stopwords = nltk_stopwords
        
        if self.nltk_stopwords:
            if language == 'en':
                self.stopwords = stopwords.words('english')
            elif language == 'fr':
                self.stopwords = stopwords.words('french')
            elif language == 'de':
                self.stopwords = stopwords.words('german')
            else:
                raise Exception("This language %s is not supported." % language)
            
        if pos_list:
            raise Exception('POS filtering is not supported yet.')
            
    def filter(self, feed_dict, replace = False):
        """
            Main method. Filter the feed_dict by creating
            two new quantities: 
                - filtered_bigrams_count
                - filterd_list_of_words
            where it respectively removed bigrams containing 
            stopwords and removing stopwords.
            If replace == True then the original feed_fict
            is replaced by the filtered one. Otherwise it just
            adds keys.
            Args:
                feed_dict [dict] containing at least bigrams_count and
                                 list_of_words 
        """
        print("\t\tFiltering...")

        assert 'bigrams_count' in feed_dict.keys()
        assert 'list_of_words' in feed_dict.keys()

        print("\t\t\tList of words...")
        filtered_list_of_words = []
        num_filtered = 0
        for k in range(len(feed_dict['list_of_words'])):
            w = feed_dict['list_of_words'][k]
            if w in self.stopwords:
                num_filtered += 1
                continue
            filtered_list_of_words.append(w)
        prop_filtered = np.round(100*num_filtered/len(feed_dict['list_of_words']), 2)
        print("\t\t\tFiltered out {} words ( {} % ).".format(num_filtered, 
                                                             prop_filtered))
        feed_dict['filtered_list_of_words'] = filtered_list_of_words

        print("\t\t\tBigrams count...")
        filtered_bigrams_count = Counter()
        num_filtered = 0
        for w1, w2 in feed_dict['bigrams_count'].keys():
            count = feed_dict['bigrams_count'][(w1, w2)]
            if w1 in self.stopwords or w2 in self.stopwords:
                num_filtered += 1
                continue
            filtered_bigrams_count[(w1,w2)] = count
        prop_filtered = np.round(100*num_filtered/len(feed_dict['bigrams_count']), 2)
        print("\t\t\tFiltered out {} bigrams ( {} % ).".format(num_filtered,
                                                               prop_filtered))
        feed_dict['filtered_bigrams_count'] = filtered_bigrams_count

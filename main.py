
import argparse
import tqdm
import numpy as np

from datasets import load_dataset

from src.utils import retrieve_text_from_wikipedia, save_analysis
from src.cooccurrences import Cooccurrences
from src.metrics import Metrics
from src.shuffle import Shuffler
from src.plot import Plotter
from src.filter import Filter


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False)
parser.add_argument('--n_articles',
                    type=int,
                    default=100,
                    dest='n_articles',
                    help='Number of articles to load from the Wikipedia dataset.')
parser.add_argument('--n_shuffle',
                    type=int,
                    default=10,
                    dest='n_shuffle',
                    help='Number of shuffled corpus will be analyzed.')
parser.add_argument('--metrics',
                    type=str,
                    default='sparsity-zipf_fit-clustering',
                    dest='metrics',
                    help='Metrics used to evaluate co-occurrences analysis. Separator "-".')
parser.add_argument('--filters',
                    type=str,
                    default='stopwords-content-function',
                    dest='filters',
                    help='Which filter to apply: stopwords, content, function. Separator "-".')
parser.add_argument('--plot',
                    action='store_true',
                    help='Plot the results.')
args = parser.parse_args()

args.metrics = args.metrics.split('-')
args.filters = args.filters.split('-')


if __name__ == '__main__':
    
    # Chose Corpus
    corpus = {'en': load_dataset("wikipedia", "20220301.en")['train'],
              'fr': load_dataset("wikipedia", "20220301.fr")['train'],
              'de': load_dataset("wikipedia", "20220301.de")['train']}
        
    ## Analysis
    print("\n\n############ Beginning of analysis ############\n\n")
    
    # Create Metrics object
    metrics = Metrics(metrics = args.metrics,
                      filters = args.filters)
    
    # Start
    analysis = {k: {} for k in corpus.keys()}

    for lang in corpus.keys():
        print("\tNow analyzing %s..\n" % lang)

        # Create the Cooccurrence and Filter object
        cooc = Cooccurrences(silent=True)
        filter = Filter(language = lang,
                        filters = args.filters)
        
        shuffler = Shuffler(language=lang)

        idx_to_load = np.random.choice(len(corpus[lang]), args.n_articles)

        print("\t### Analyzing vanilla data ###\n")
        # Adding vanilla data to the cooc
        print("\t\tRetrieving text from Wikipedia Dataset..")
        cooc.update_text(
                text = retrieve_text_from_wikipedia(
                        wiki_data = corpus[lang], 
                        idx = idx_to_load 
                        )
                )
        # /!\ text is supposedly HUGE /!\
        # store its size:
        print("\t\tTotal text size: %s caracters."%len(cooc.text))

        # Compute POS Sequence and Es sets
        print("\t\tCompute POS sequence and Es sets...")
        shuffler.compute_pos_sequence_and_E_sets(cooc.text)

        # Compute feed_forward dict
        feed_dict = cooc.compute_feed_dict(bigram_counts=True,
                                           list_of_words=True)

        # Filter out some words
        filter.filter(feed_dict)

        # Compute Metrics
        analysis_lang = metrics.compute_metrics(feed_dict=feed_dict)

        analysis[lang]['vanilla'] = analysis_lang
        print("\t\tDone!")
        
        ## BootStrapping

        # Token Level Shuffling
        print("\n\t### Analyzing token level shuffled corpus ###\n")
        for i in range(args.n_shuffle):
            print("\t\tShuffling Dataset..")
            cooc.update_text(
                text = shuffler.token_shuffle(cooc.text),
                just_shuffled = True
                )
        
            # Compute feed_forward dict
            feed_dict = cooc.compute_feed_dict(bigram_counts=True,
                                               list_of_words=True)
            
            filter.filter(feed_dict)

            analysis_lang = metrics.compute_metrics(feed_dict=feed_dict)

            analysis[lang]['token_shuffled_%s' % i] = analysis_lang

        # POS Level Shuffling
        print("\n\t### Analyzing shuffled corpus w.r.t. POS ###\n")
        for i in range(args.n_shuffle):
            print("\t\tShuffling Dataset..")
            cooc.update_text(
                text = shuffler.pos_shuffle(cooc.text),
                just_shuffled = True
                )
            # Compute feed_forward dict
            feed_dict = cooc.compute_feed_dict(bigram_counts=True,
                                               list_of_words=True)
            
            filter.filter(feed_dict)

            analysis_lang = metrics.compute_metrics(feed_dict=feed_dict)

            analysis[lang]['pos_shuffled_%s' % i] = analysis_lang
        
        
    ### End of Analysis ##
    print("\n\n############ End of analysis ############\n\n")

    ### Save

    save_analysis(analysis = analysis,
                  languages = corpus.keys(),
                  n_articles = args.n_articles,
                  n_shuffle = args.n_shuffle)

    ### Plot

    if args.plot:
        plotter = Plotter(analysis = analysis,
                          args = args)
        plotter.plot()

        for filter in args.filters:
            plotter.set_filtered(filter)
            plotter.plot()
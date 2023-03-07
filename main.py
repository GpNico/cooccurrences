
import argparse
import tqdm
import numpy as np

from datasets import load_dataset

from src.utils import retrieve_text_from_wikipedia
from src.cooccurrences import Cooccurrences
from src.metrics import Metrics
from src.shuffle import Shuffler
from src.plot import Plotter


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
args = parser.parse_args()

args.metrics = args.metrics.split('-')


if __name__ == '__main__':
    
    # Chose Corpus
    corpus = {'en': load_dataset("wikipedia", "20220301.en")['train'],}
              #'fr': load_dataset("wikipedia", "20220301.fr")['train'],
              #'de': load_dataset("wikipedia", "20220301.de")['train']}
        
    ## Analysis
    print("\n\n############ Beginning of analysis ############\n\n")
    
    # Create Metrics and Shuffler objects
    metrics = Metrics(metrics = args.metrics)
    
    # Start
    analysis = {k: {} for k in corpus.keys()}

    for lang in corpus.keys():
        print("\tNow analyzing %s..\n" % lang)

        # Create the Cooccurrence and Metrics object
        cooc = Cooccurrences(silent=True)
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

        # Compute feed_forward dict
        feed_dict = cooc.compute_feed_dict(bigram_counts=True,
                                           list_of_words=True)

        analysis_lang = metrics.compute_metrics(feed_dict=feed_dict)

        analysis[lang]['vanilla'] = analysis_lang
        print("\t\tDone!")
        
        ## BootStrapping
        shuffler = Shuffler(language=lang)

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

            analysis_lang = metrics.compute_metrics(feed_dict=feed_dict)

            analysis[lang]['token_shuffled_%s' % i] = analysis_lang

        # POS Level Shuffling
        print("\n\t### Analyzing shuffled corpus w.r.t. POS ###\n")
        # Need to reset text
        cooc.update_text(
                text = retrieve_text_from_wikipedia(
                        wiki_data = corpus[lang], 
                        idx = idx_to_load 
                        )
                )
        for i in range(args.n_shuffle):
            print("\t\tShuffling Dataset..")
            cooc.update_text(
                text = shuffler.pos_shuffle(cooc.text),
                just_shuffled = True
                )
            # Compute feed_forward dict
            feed_dict = cooc.compute_feed_dict(bigram_counts=True,
                                               list_of_words=True)

            analysis_lang = metrics.compute_metrics(feed_dict=feed_dict)

            analysis[lang]['pos_shuffled_%s' % i] = analysis_lang
        
        
    ### End of Analysis ##
    print("\n\n############ End of analysis ############\n\n")

    ### Plot

    plotter = Plotter(analysis = analysis,
                      metrics = args.metrics)
    plotter.plot()
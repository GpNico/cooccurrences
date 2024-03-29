
import argparse
import tqdm
import numpy as np

from src.utils import retrieve_text_from_wikipedia, retrieve_text_from_oscar, save_analysis, from_dict_to_csv
from src.cooccurrences import Cooccurrences
from src.metrics import Metrics
from src.shuffle import Shuffler
from src.plot import Plotter
from src.filter import Filter
from config import languages2compute


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False)
parser.add_argument('--size',
                    type=int,
                    default=10e6,
                    dest='size',
                    help='Number of caracters that constitute the dataset.')
parser.add_argument('--n_shuffle',
                    type=int,
                    default=10,
                    dest='n_shuffle',
                    help='Number of shuffled corpus will be analyzed.')
parser.add_argument('--metrics',
                    type=str,
                    default='sparsity-zipf_fit',
                    dest='metrics',
                    help='Metrics used to evaluate co-occurrences analysis. Separator "-".')
parser.add_argument('--filters',
                    type=str,
                    default='stopwords-content-function',
                    dest='filters',
                    help='Which filter to apply: stopwords, content, function. Separator "-".')
parser.add_argument('--dataset',
                    type=str,
                    default='oscar',
                    dest='dataset',
                    help='We have wikipedia with only en, fr, de. And we have oscar with all spacy languages.')
parser.add_argument('--plot',
                    action='store_true',
                    help='Plot the results.')
parser.add_argument('--csv',
                    action='store_true',
                    help='Convert dict results into csv')
args = parser.parse_args()

args.metrics = args.metrics.split('-')
args.filters = args.filters.split('-')


if __name__ == '__main__':
    
    languages = [lang for lang, b in languages2compute.items() if b]
    
    ## Only convert the results to csv
    if args.csv:
        from_dict_to_csv(languages = languages,
                         size = args.size,
                         n_shuffle = args.n_shuffle,
                         dataset = args.dataset,
                         filters = args.filters)
        exit(0)
    
    ## Analysis
    print("\n\n############ Beginning of analysis ############\n")
    
    # Create Metrics object
    metrics = Metrics(metrics = args.metrics,
                      filters = args.filters)
    
    # Start
    analysis = {k: {} for k in languages}

    for lang in languages:
        print("\n\tNow analyzing %s..\n" % lang)

        # Create the Cooccurrence and Filter object
        cooc = Cooccurrences(language = lang,
                             silent=True)
        filter = Filter(language = lang,
                        filters = args.filters)
        
        shuffler = Shuffler(language=lang)

        print("\t### Analyzing vanilla data ###\n")
        # Adding vanilla data to the cooc
        print(f"\t\tRetrieving text from {args.dataset} dataset..")
        if args.dataset == "wikipedia":
            text_size, text, articles_idx = retrieve_text_from_wikipedia(
                                    language = lang,
                                    size = args.size
                                    )
        elif args.dataset == "oscar":
            text_size, text, articles_idx = retrieve_text_from_oscar(
                                    language = lang,
                                    size = args.size
                                    )
        else:
            raise Exception(f"No such dataset as {args.dataset}.")
        
        cooc.update_text(
                text = text
                )
        text = None # No need to keep it in memory
        # /!\ text is supposedly HUGE /!\

        # store its size:
        print("\t\tTotal text size: %s caracters."%text_size)
        analysis[lang]['size'] = text_size

        # Compute POS Sequence and Es sets
        print("\t\tCompute POS sequence and Es sets...")
        shuffler.compute_pos_sequence_and_E_sets(cooc.text,
                                                 articles_idx = articles_idx,
                                                 n_subsets = 50)

        # Compute feed_forward dict
        feed_dict = cooc.compute_feed_dict()

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
            feed_dict = cooc.compute_feed_dict()
            
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
            feed_dict = cooc.compute_feed_dict()
            
            filter.filter(feed_dict)

            analysis_lang = metrics.compute_metrics(feed_dict=feed_dict)

            analysis[lang]['pos_shuffled_%s' % i] = analysis_lang
         
        # We save for each language to allow early termination    
        save_analysis(analysis = analysis,
                      language = lang,
                      size = args.size,
                      n_shuffle = args.n_shuffle,
                      dataset = args.dataset)
        
        
    ### End of Analysis ##
    print("\n\n############ End of analysis ############\n\n")

    ### Create CSV
    
    from_dict_to_csv(res_dict = analysis,
                     size = args.size,
                     n_shuffle = args.n_shuffle,
                     dataset = args.dataset,
                     filters = args.filters)

    ### Plot

    if args.plot:
        plotter = Plotter(analysis = analysis,
                          args = args)
        plotter.plot()

        for filter in args.filters:
            plotter.set_filtered(filter)
            plotter.plot()
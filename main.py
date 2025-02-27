
import argparse
import tqdm
import numpy as np
import pickle
import os
import time

from src.utils import retrieve_text_from_wikipedia, retrieve_text_from_oscar, save_analysis, from_dict_to_csv, get_save_folder_name
from src.cooccurrences import Cooccurrences
from src.metrics import Metrics
from src.shuffle import Shuffler
from src.plot import Plotter
from src.filter import Filter
from config import languages2compute


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False)
parser.add_argument('--size',
                    type=int,
                    default=1e6, #5*1e7
                    dest='size',
                    help='Number of caracters that constitute the dataset.')
parser.add_argument('--n_shuffle',
                    type=int,
                    default=2, #10
                    dest='n_shuffle',
                    help='Number of shuffled corpus will be analyzed.')
parser.add_argument('--split',
                    type=str,
                    default='sentence', # Should try both
                    dest='split',
                    help='Do we compute cooccurrences based on a sliding window or sentences.')
parser.add_argument('--ordered', # Shoulc Always be True as we can always symmetrize a matrix
                    type=bool,
                    default=True,
                    dest="ordered",
                    help='Should the pairs be ordered ie "a b" has only (a,b) as cooccurrence not (b, a).')
parser.add_argument('--separate_freq_rank', # Shoulc Always be True as we can always symmetrize a matrix
                    type=bool,
                    default=True,
                    dest="sep",
                    help='If True the frequencies and ranks are computed from separate corpora.')
parser.add_argument('--window_size',
                    type=int,
                    default=5,
                    dest='window_size',
                    help='Size of the slidding window.')
parser.add_argument('--metrics', # We will compute it after so -> "None"
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
parser.add_argument('--language',
                    type=str,
                    default='None',
                    dest='language',
                    help='Which language to compute. If None then use config file.')
parser.add_argument("--analysis", # No Need 
                    action = 'store_true',
                    help = "If True then metrics & filter are computed.")
parser.add_argument('--plot', # No need
                    action='store_true',
                    help='Plot the results.')
parser.add_argument('--csv', # No need
                    action='store_true',
                    help='Convert dict results into csv')
parser.add_argument('--jean_zay', # No need
                    action='store_true',
                    help='If on Jean Zay load OSCAR from disk.')
args = parser.parse_args()

args.metrics = args.metrics.split('-')
args.filters = args.filters.split('-')


if __name__ == '__main__':
    
    time_beg = time.time()
    
    exps = ['']
    if args.sep:
        exps.append('_ranks')
        
    for exp in exps:
        # Raws folder
        raws_path = get_save_folder_name(
                        language=None,
                        size = args.size,
                        n_shuffle = args.n_shuffle,
                        dataset = args.dataset,
                        split = args.split,
                        ordered = args.ordered,
                        window_size = args.window_size)

        raws_path = os.path.join('results', 'raws', raws_path)
        
        os.makedirs(
            raws_path,
            exist_ok = True
        )
        
        if args.language == 'None':
            languages = [lang for lang, b in languages2compute.items() if b]
        else:
            languages = [args.language]
        
        ## Only convert the results to csv
        if args.csv:
            from_dict_to_csv(languages = languages,
                            size = args.size,
                            n_shuffle = args.n_shuffle,
                            dataset = args.dataset,
                            filters = args.filters,
                            split = args.split,
                            ordered = args.ordered,
                            window_size = args.window_size)
            exit(0)
        
        ####### Analysis #######
        print("\n\n############ Beginning of analysis ############\n")
        
        # Create Metrics object
        metrics = Metrics(metrics = args.metrics,
                        filters = args.filters)
        
        # Start
        if args.analysis:
            analysis = {k: {} for k in languages}

        for lang in languages:
            print("\n\tNow analyzing %s..\n" % lang)

            # Create the Cooccurrence and Filter object
            cooc = Cooccurrences(language = lang,
                                split = args.split,
                                ordered = args.ordered,
                                window_size = args.window_size,
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
                                        size = args.size,
                                        jean_zay = args.jean_zay
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
            if args.analysis:
                analysis[lang]['size'] = text_size

            # Compute POS Sequence and Es sets
            print("\t\tCompute POS sequence and Es sets...")
            shuffler.compute_pos_sequence_and_E_sets(cooc.text,
                                                    articles_idx = articles_idx,
                                                    n_subsets = 50)

            ### Vanilla ###

            # Compute feed_forward dict
            feed_dict = cooc.compute_feed_dict()
            
            # Here save Feed-Dict
            with open(os.path.join(raws_path, f'{lang}_vanilla{exp}.pickle'), 'wb') as handle:
                pickle.dump(feed_dict, 
                            handle, 
                            protocol=pickle.HIGHEST_PROTOCOL)

            if args.analysis:
                # Filter out some words
                filter.filter(feed_dict)

                # Compute Metrics
                analysis_lang = metrics.compute_metrics(feed_dict=feed_dict)

                analysis[lang]['vanilla'] = analysis_lang
                print("\t\tDone!")
            
            ##### BootStrapping #####

            ## Token Level Shuffling ##
            print("\n\t### Analyzing token level shuffled corpus ###\n")
            for i in range(args.n_shuffle):
                print("\t\tShuffling Dataset..")
                cooc.update_text(
                    text = shuffler.token_shuffle(cooc.text),
                    just_shuffled = True
                    )
            
                # Compute feed_forward dict
                feed_dict = cooc.compute_feed_dict()
                
                with open(os.path.join(raws_path, f'{lang}_token_{i}{exp}.pickle'), 'wb') as handle:
                    pickle.dump(feed_dict, 
                                handle, 
                                protocol=pickle.HIGHEST_PROTOCOL)
                
                
                if args.analysis:
                    filter.filter(feed_dict)

                    analysis_lang = metrics.compute_metrics(feed_dict=feed_dict)

                    analysis[lang]['token_shuffled_%s' % i] = analysis_lang

            ## POS Level Shuffling ##
            print("\n\t### Analyzing shuffled corpus w.r.t. POS ###\n")
            for i in range(args.n_shuffle):
                print("\t\tShuffling Dataset..")
                cooc.update_text(
                    text = shuffler.pos_shuffle(cooc.text),
                    just_shuffled = True
                    )
                # Compute feed_forward dict
                feed_dict = cooc.compute_feed_dict()
                
                with open(os.path.join(raws_path, f'{lang}_pos_{i}{exp}.pickle'), 'wb') as handle:
                    pickle.dump(feed_dict, 
                                handle, 
                                protocol=pickle.HIGHEST_PROTOCOL)
                
                if args.analysis:
                    filter.filter(feed_dict)

                    analysis_lang = metrics.compute_metrics(feed_dict=feed_dict)

                    analysis[lang]['pos_shuffled_%s' % i] = analysis_lang
            
            
            ##### End of Boostrapping #####
            
            
            if args.analysis:
                # We save for each language to allow early termination    
                save_analysis(analysis = analysis,
                            language = lang,
                            size = args.size,
                            n_shuffle = args.n_shuffle,
                            dataset = args.dataset,
                            split = args.split,
                            ordered = args.ordered,
                            window_size = args.window_size)
            
            
        ####### End of Analysis #######
        print("\n\n############ End of analysis ############\n\n")

    ### Create CSV
    if args.analysis:
        from_dict_to_csv(res_dict = analysis,
                        size = args.size,
                        n_shuffle = args.n_shuffle,
                        dataset = args.dataset,
                        filters = args.filters,
                        split = args.split,
                        ordered = args.ordered,
                        window_size = args.window_size)

    ### Plot
    
    if args.analysis and args.plot:
        plotter = Plotter(analysis = analysis,
                          args = args)
        plotter.plot()

        for filter in args.filters:
            plotter.set_filtered(filter)
            plotter.plot()
            
    elapsed_time = time.time() - time_beg
    # Formatting the output dynamically
    if elapsed_time < 60:
        formatted_time = f"{elapsed_time:.2f} seconds"
    elif elapsed_time < 3600:
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        formatted_time = f"{minutes} min {seconds} sec"
    else:
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        formatted_time = f"{hours} hr {minutes} min {seconds} sec"
    print(f"Execution Time: {formatted_time}")
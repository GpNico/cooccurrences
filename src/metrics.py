import numpy as np

from sklearn.linear_model import LinearRegression
import networkx as nx


class Metrics:
    """
        Wrapper class to compute all co-occurrences metrics.
        
        The main method is compute_metrics. It takes in argument a feed_dict 
        that is supposed to contain all necessary materials.

        All _compute_XX methods return a metrics_dict containing relevant values
        and each of its key begins by XX_ to identify more easily the metrics in
        postprocessing (like plotting). 
    """

    def __init__(self, metrics = ['sparsity', 'zipf_fit']):
        # metrics to evaluate
        self.metrics = metrics

    def compute_metrics(self, feed_dict):
        """
            Run all metrics.
            Args:
                feed_dict [dict]
            Returns:
                metrics_dict [dict] 
        """
        print("\t\tComputing Metrics...")

        # Compute metrics
        metrics_dict = {}
        if 'sparsity' in self.metrics:
            print('\t\t\tSparsity...')
            metrics_dict.update(
                        self._compute_sparsity(
                                num_bigrams = len(feed_dict['bigrams_count']),
                                num_words = len(feed_dict['list_of_words'])
                                )
                        )
        if 'zipf_fit' in self.metrics:
            print('\t\t\tZipf Fit...')
            metrics_dict.update(
                        self._compute_zipf_fit(
                                bigrams_count = feed_dict['bigrams_count']
                                )
                        )
        if 'clustering' in self.metrics:
            print('\t\t\tClustering...')
            metrics_dict.update(
                        self._compute_clustering(
                                bigrams = list(feed_dict['bigrams_count'].keys())
                                )
                        )

        return metrics_dict


    def _compute_sparsity(self, num_bigrams, num_words):
        """
            Compute the sparsity metrics dict.
            Args:
                num_bigrams [int]
                num_words [int]
            Return:
                metric_dict [dict]
        """
        num_cells = num_words**2
        return {'sparsity_nonzeros': num_bigrams,
                'sparsity_zeros': num_cells - num_bigrams,
                'sparsity_prop_nonzeros': num_bigrams/num_cells,
                'sparsity_prop_zeros': (num_cells - num_bigrams)/num_cells}  

    def _compute_zipf_fit(self, bigrams_count, eps = 1e-8):
        """
            Compute k in the Zipfs law as well as the R^2.
            Args:
                bigram_counts []
                eps [float] for regularization purpose
            Return:
                metric_dict [dict]
        """
        values = np.flip(np.sort(np.array(list(bigrams_count.values())).flatten()))
        ranks = np.arange(1, len(values)+ 1)
        
        # Regression
        X = np.expand_dims(np.log(ranks), axis = 1)
        y = np.log(values + eps) # reg
        reg = LinearRegression().fit(X, y)
        a, b = reg.coef_, reg.intercept_
        Rsq = reg.score(X,y)

        return {'zipf_fit_coeff': a[0],
                'zipf_fit_intercept': b, 
                'zipf_fit_R_sq':Rsq}
    
    def _compute_clustering(self, bigrams):
        """
            Create a graph from a list of bigrams. And compute its
            average clustering coefficient and its global clustering
            coefficient (or transitivity).
            Args:
                bigrams [list of tuple] ex: [('a', 'a'), ('a', 'b')]
        """
        G = nx.Graph()
        G.add_edges_from(bigrams)
        return {'clustering_average': 0, #nx.average_clustering(G),
                'clustering_transitivity': nx.transitivity(G)}

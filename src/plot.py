
import os
import matplotlib.pyplot as plt
import numpy as np

from src.utils import from_metrics2compute_to_metrics2plot


class Plotter:
    """
        Wrapper Class to plot different metrics.
    """

    def __init__(self, analysis, args):
        """
            Args:
                analysis [dict] keys [str] language
                                values [dict] keys [str] vanilla/shuffle
                                              values [dict] keys [str] keys metrics
                                                            values [float]
                args [argparse]
        """
        self.analysis = analysis
        self.metrics2compute = args.metrics
        self.metrics2plot = from_metrics2compute_to_metrics2plot(self.metrics2compute)

        # For naming
        self.size = args.size
        self.n_shuffle = args.n_shuffle
        self.dataset = args.dataset

        # Filtered flag (no filter at iniziatilation)
        self.set_filtered('')

        # Init things (!)
        self._get_languages()
        self._get_conditions()
        self._get_legend()
        self._get_bar_offsets()
        self._preprocess_saving()

        # Plot params
        self.n_plots = len(self.metrics2compute)
        self.plot_sizes = (5,5)
        self.color_vanilla = 'royalblue'
        self.color_token_shuffled = 'darkmagenta'
        self.color_pos_shuffled = 'red'
        
    def plot(self):
        """
            Main method of the class.
            Compute and save a plot PER metrics to compute.
            This means that the sparsity metric will be composed of 1 subplot
            where the zipf_fit will be composed of two.
        """
        if 'sparsity' in self.metrics2compute:
            self._plot_sparsity()
        if 'zipf_fit' in self.metrics2compute:
            self._plot_zipf_fit()
        if 'clustering' in self.metrics2compute:
            self._plot_clustering()

    def _plot_sparsity(self):
        metrics = self.metrics2plot['sparsity']
        assert len(metrics) == 1

        _, ax = plt.subplots(1, 1, figsize = (self.plot_sizes[0],
                                              self.plot_sizes[1]))

        for l_idx, l in enumerate(self.languages):
            # Get data
            vanilla_data = []
            token_shuffled_data = []
            pos_shuffled_data = []
            for cond in self.conditions:
                if 'vanilla' in cond:
                    vanilla_data.append(
                        self.analysis[l][cond][self.filtered_flag + metrics[0]]
                        )
                elif 'token_shuffled' in cond:
                    token_shuffled_data.append(
                        self.analysis[l][cond][self.filtered_flag + metrics[0]]
                        )
                elif 'pos_shuffled' in cond:
                    pos_shuffled_data.append(
                        self.analysis[l][cond][self.filtered_flag + metrics[0]]
                        )
                else:
                    if cond == 'size':
                        continue
                    raise Exception("Unexpected condition %s" % cond)
            # Plot
            if 'vanilla' in self.conditions:
                x_vanilla = l_idx*np.ones(len(vanilla_data)) + self.bar_offsets[0]
                ax.scatter(x_vanilla, vanilla_data, color = self.color_vanilla)
                ax.bar(x_vanilla, vanilla_data, width = self.bar_width, color = self.color_vanilla, alpha = 0.5)
            if 'token_shuffled_0' in self.conditions:
                x_token_shuffled = l_idx*np.ones(len(token_shuffled_data)) + self.bar_offsets[1]
                #ax.scatter(x_token_shuffled[0], np.mean(token_shuffled_data), color = self.color_token_shuffled)
                ax.scatter(x_token_shuffled, token_shuffled_data, color = self.color_token_shuffled)
                ax.bar(x_token_shuffled[0], np.mean(token_shuffled_data), width = self.bar_width, color = self.color_token_shuffled, alpha = 0.5)
            if 'pos_shuffled_0' in self.conditions:
                x_pos_shuffled = l_idx*np.ones(len(pos_shuffled_data)) + self.bar_offsets[2]
                ax.scatter(x_pos_shuffled, pos_shuffled_data, color = self.color_pos_shuffled)
                ax.bar(x_pos_shuffled[0], np.mean(pos_shuffled_data), width = self.bar_width, color = self.color_pos_shuffled, alpha = 0.5)
            
        ax.set_title('Sparsity')
        ax.set_xticks(np.arange(len(self.languages)) + 0.5)
        ax.set_xticklabels(self.languages)
        ax.set_axisbelow(True)
        ax.legend(self.legend, loc = 'lower right')
        ax.grid()
    
        plt.savefig(self.figname_template % (self.filtered_flag + 'sparsity'))

    def _plot_zipf_fit(self):
            metrics = self.metrics2plot['zipf_fit']
            assert len(metrics) == 2

            _, axs = plt.subplots(1, 2, figsize = (2*self.plot_sizes[0],
                                                   self.plot_sizes[1]))

            for k in range(len(metrics)):
                for l_idx, l in enumerate(self.languages):
                    # Get data
                    vanilla_data = []
                    token_shuffled_data = []
                    pos_shuffled_data = []
                    for cond in self.conditions:
                        if 'vanilla' in cond:
                            vanilla_data.append(
                                self.analysis[l][cond][self.filtered_flag + metrics[k]]
                                )
                        elif 'token_shuffled' in cond:
                            token_shuffled_data.append(
                                self.analysis[l][cond][self.filtered_flag + metrics[k]]
                                )
                        elif 'pos_shuffled' in cond:
                            pos_shuffled_data.append(
                                self.analysis[l][cond][self.filtered_flag + metrics[k]]
                                )
                        else:
                            if cond == 'size':
                                continue
                            raise Exception("Unexpected condition %s" % cond)
                    # Plot
                    if 'vanilla' in self.conditions:
                        x_vanilla = l_idx*np.ones(len(vanilla_data)) + self.bar_offsets[0]
                        axs[k].scatter(x_vanilla, vanilla_data, color = self.color_vanilla)
                        axs[k].bar(x_vanilla, vanilla_data, width = self.bar_width, color = self.color_vanilla, alpha = 0.5)
                    if 'token_shuffled_0' in self.conditions:
                        x_token_shuffled = l_idx*np.ones(len(token_shuffled_data)) + self.bar_offsets[1]
                        #ax.scatter(x_token_shuffled[0], np.mean(token_shuffled_data), color = self.color_token_shuffled)
                        axs[k].scatter(x_token_shuffled, token_shuffled_data, color = self.color_token_shuffled)
                        axs[k].bar(x_token_shuffled[0], np.mean(token_shuffled_data), width = self.bar_width, color = self.color_token_shuffled, alpha = 0.5)
                    if 'pos_shuffled_0' in self.conditions:
                        x_pos_shuffled = l_idx*np.ones(len(pos_shuffled_data)) + self.bar_offsets[2]
                        axs[k].scatter(x_pos_shuffled, pos_shuffled_data, color = self.color_pos_shuffled)
                        axs[k].bar(x_pos_shuffled[0], np.mean(pos_shuffled_data), width = self.bar_width, color = self.color_pos_shuffled, alpha = 0.5)

                if metrics[k] == 'zipf_fit_R_sq':   
                    axs[k].set_title('Zipf Fit: R^2')
                elif metrics[k] == 'zipf_fit_coeff':
                    axs[k].set_title('Zipf Fit: Coeff.')
                axs[k].set_xticks(np.arange(len(self.languages)) + 0.5)
                axs[k].set_xticklabels(self.languages)
                axs[k].set_axisbelow(True)
                axs[k].legend(self.legend, loc = 'lower right')
                axs[k].grid()
        
            plt.savefig(self.figname_template % (self.filtered_flag + 'zipf_fit'))


    def _plot_clustering(self):
            metrics = self.metrics2plot['clustering']
            assert len(metrics) == 2

            _, axs = plt.subplots(1, 2, figsize = (2*self.plot_sizes[0],
                                                   self.plot_sizes[1]))

            for k in range(len(metrics)):
                for l_idx, l in enumerate(self.languages):
                    # Get data
                    vanilla_data = []
                    token_shuffled_data = []
                    pos_shuffled_data = []
                    for cond in self.conditions:
                        if 'vanilla' in cond:
                            vanilla_data.append(
                                self.analysis[l][cond][self.filtered_flag + metrics[k]]
                                )
                        elif 'token_shuffled' in cond:
                            token_shuffled_data.append(
                                self.analysis[l][cond][self.filtered_flag + metrics[k]]
                                )
                        elif 'pos_shuffled' in cond:
                            pos_shuffled_data.append(
                                self.analysis[l][cond][self.filtered_flag + metrics[k]]
                                )
                        else:
                            if cond == 'size':
                                continue
                            raise Exception("Unexpected condition %s" % cond)
                    # Plot
                    if 'vanilla' in self.conditions:
                        x_vanilla = l_idx*np.ones(len(vanilla_data)) + self.bar_offsets[0]
                        axs[k].scatter(x_vanilla, vanilla_data, color = self.color_vanilla)
                        axs[k].bar(x_vanilla, vanilla_data, width = self.bar_width, color = self.color_vanilla, alpha = 0.5)
                    if 'token_shuffled_0' in self.conditions:
                        x_token_shuffled = l_idx*np.ones(len(token_shuffled_data)) + self.bar_offsets[1]
                        #ax.scatter(x_token_shuffled[0], np.mean(token_shuffled_data), color = self.color_token_shuffled)
                        axs[k].scatter(x_token_shuffled, token_shuffled_data, color = self.color_token_shuffled)
                        axs[k].bar(x_token_shuffled[0], np.mean(token_shuffled_data), width = self.bar_width, color = self.color_token_shuffled, alpha = 0.5)
                    if 'pos_shuffled_0' in self.conditions:
                        x_pos_shuffled = l_idx*np.ones(len(pos_shuffled_data)) + self.bar_offsets[2]
                        axs[k].scatter(x_pos_shuffled, pos_shuffled_data, color = self.color_pos_shuffled)
                        axs[k].bar(x_pos_shuffled[0], np.mean(pos_shuffled_data), width = self.bar_width, color = self.color_pos_shuffled, alpha = 0.5)

                if metrics[k] == 'clustering_average':   
                    axs[k].set_title('Clustering: Average')
                elif metrics[k] == 'clustering_transitivity':
                    axs[k].set_title('Clustering: Transitivity')
                axs[k].set_xticks(np.arange(len(self.languages)) + 0.5)
                axs[k].set_xticklabels(self.languages)
                axs[k].set_axisbelow(True)
                axs[k].legend(self.legend, loc = 'lower right')
                axs[k].grid()
        
            plt.savefig(self.figname_template % (self.filtered_flag + 'clustering'))


    def _preprocess_saving(self):
        """
            Do all the boring things concerning saving:
                - creates "results folder"
                - creates templae for file saving
        """
        # Names and related matter
        self.figname_template = os.path.join("results", 
                                             "plots")
        os.makedirs(self.figname_template, exist_ok=True)
        self.figname_template = os.path.join(self.figname_template, 
                                             self.dataset)
        
        for lang in self.languages:
            self.figname_template += '_%s'%lang
        self.figname_template += '_size_%s_n_shuffle_%s_' % (self.size,
                                                                   self.n_shuffle)
        self.figname_template += "%s.png" # to add metrics name later on  

    def _get_languages(self):
        """
            Creates languages attribute: [List of str] ex: ['en', 'fr']
        """
        self.languages = list(self.analysis.keys())

    def _get_conditions(self):
        """
            Creates conditions atrtributes: [List of str] ex: ['vanilla', 'token_shuffled', 'pos_shuffled']
        """
        self.conditions = list(self.analysis[self.languages[0]].keys())
        self.n_cond = 0

    def _get_legend(self):
        self.legend = []
        if 'vanilla' in self.conditions:
            self.legend.append('vanilla')
            self.n_cond += 1
        if 'token_shuffled_0' in self.conditions:
            self.legend.append('token shuffled')
            self.n_cond += 1
        if 'pos_shuffled_0' in self.conditions:
            self.legend.append('POS shuffled')
            self.n_cond += 1

    def _get_bar_offsets(self):
        if self.n_cond == 1:
            self.bar_offsets = [0.5]
            self.bar_width = 0.8
        elif self.n_cond == 2:
            self.bar_offsets = [0.3, 0.7]
            self.bar_width = 0.4
        elif self.n_cond == 3:
            self.bar_offsets = [0.26, 0.5, 0.74]
            self.bar_width = 0.24

    def set_filtered(self, filter: str):
        if filter == '':
            self.filtered_flag = ''
        else:
            self.filtered_flag = 'filtered_%s_' % filter


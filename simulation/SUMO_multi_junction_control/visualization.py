import matplotlib.pyplot as plt
import os
import pandas as pd

class Visualization:
    def __init__(self, path, dpi):
            self._path = path
            self._dpi = dpi

    def save_local_performance_as_csv(self, data, filename):
        """
        Saves information about an agent's local performance to csv

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        filename : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        data = {'Local reward': data[0],
                   'Cumulative delay': data[1],
                   'Average local queue length': data[2]}
        df = pd.DataFrame(data, columns = ['Local reward', 'Cumulative delay', 'Average local queue length'])
        df.to_csv(self._path + "/" + filename)
        
        
    def save_global_statistics_as_csv(self,t_test_cum_wait, t_test_queue_length):
        """
        Generating .csv file of global statistics

        Parameters
        ----------
        t_test_cum_wait : TYPE
            DESCRIPTION.
        t_test_queue_length : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        statistics_data = {'t-statistic cumulative wait': [x.statistic for x in t_test_cum_wait],
                   'pvalue cumulative wait': [x.pvalue for x in t_test_cum_wait],
                   't-statistic queue length': [x.statistic for x in t_test_queue_length],
                   'pvalue queue length': [x.pvalue for x in t_test_queue_length]}
        stats_df = pd.DataFrame(statistics_data, columns = ['t-statistic cumulative wait', 'pvalue cumulative wait', 
                                                 't-statistic queue length', 'pvalue queue length'])
        stats_df.to_csv(self._path + "/global_statistics.csv")
    
    def plot_multiple_agents(self, data, filename, xlabel, ylabel, agents):
        """
        Produce a plot of global and individual agent performance over session and save the relative data to txt
        """
        min_val = min(min(data))
        max_val = max(max(data))

        plt.rcParams.update({'font.size': 24})  # set bigger font size
        plt.title(f"{ylabel} for {agents[1:]}")
        for num, datum in enumerate(data):
            plt.plot(datum, label=agents[num])
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.margins(0)
        plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))
        plt.legend()
        fig = plt.gcf()
        fig.set_size_inches(20, 11.25)
        fig.savefig(os.path.join(self._path, 'plot_'+filename+'.png'), dpi=self._dpi)
        plt.close("all")
            

    def plot_single_agent(self, data, filename, xlabel, ylabel, agent):
        """
        Produce a plot of performance of a single agent over the session and
        include agent parameters
        """
        min_val = min(data)
        max_val = max(data)

        plt.rcParams.update({'font.size': 24})  # set bigger font size
        plt.title(f"{ylabel} for {agent}")
        plt.plot(data, label=agent)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.margins(0)
        plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))
        
        # textstr = str(parameters)
        # plt.subplots_adjust(left=0.25)
        
        fig = plt.gcf()
        fig.set_size_inches(20, 11.25)
        fig.savefig(os.path.join(self._path, 'plot_'+filename+'.png'), dpi=self._dpi)
        plt.close("all")
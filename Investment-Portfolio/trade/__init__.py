import os
import pandas as pd
import numpy as np
from data_load.stocks import Stocks
from trade.portfolio import Portfolio
import datetime
from trade.visulization import plot_cumulative_wealth


dataset_name = ["ff25_csv", "ff49_csv", "ff100_csv", "sandp_csv", "ETFs_csv"]
if __name__ == "__main__":
    PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    for dataset in dataset_name:
        file_path = PARENT_DIR + '/result/' + dataset + '.csv'
        return_list = pd.read_csv(file_path)
        plot_cumulative_wealth(return_list, dataset)

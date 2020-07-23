# coding=utf-8
import os
from data_load.stocks import Stocks

if __name__ == "__main__":

    PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mat_data_path = PARENT_DIR + '/data/ff25_csv/portfolio25.csv'
    ff25 = Stocks(path=mat_data_path, type="csv")
    print("this is main")

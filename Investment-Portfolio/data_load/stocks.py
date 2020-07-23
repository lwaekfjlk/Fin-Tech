#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    File Name:      stocks.py
    Created Time:   10/21/18 3:56 PM
    Author:         zmy
"""
import scipy.io as sio
import pandas as pd
import numpy as np
import os


class Stocks:
    def __init__(self, dict_path="", type="MAT", return_file="", is_context=False):
        """
        :param dict_path: 数据集的文档路径
        :param type: 是否是mat
        :param return_file: return文件的路径
        :param is_context: 是否要获取feature
        """
        self.path = dict_path + '/' + return_file
        self.dict_path = dict_path
        self.is_context = is_context
        if type == "MAT":
            self.Nportfolios = self.mat2dataframe(keyword="Nportfolios")[0][0]

            self.Nmonths = self.mat2dataframe(keyword="Nmonths")[0][0]
            self.portfolios = self.mat2dataframe(keyword="portfolios")
            self.portfolios = self.portfolios.values
            self.portfolios_price = self.mat2dataframe(keyword="portfolios_price")
            self.portfolios_price = self.portfolios_price.values
            self.init_time = None
        elif type == "csv":
            self.Nportfolios = self.csv2dataframe(keyword="Nportfolios")
            self.Nmonths = self.csv2dataframe(keyword="Nmonths")
            self.portfolios = self.csv2dataframe(keyword="portfolios")
            self.portfolios_price = self.csv2dataframe(keyword="portfolios_price")
            self.init_time = self.csv2dataframe(keyword="init_time")
        elif type == "txt":
            self.Nportfolios, self.Nmonths,  self.portfolios, self.portfolios_price = self.txt2dataframe()
            self.init_time = 0
        if is_context:
            self.stock_feature, self.stock_feature_dimension = self.get_stock_feature()
            self.market_feature, self.market_feature_dimension = self.get_market_feature()

    def mat2dataframe(self, keyword):
        mat_data = sio.loadmat(self.path)
        version = str(mat_data.get("__version__", "1.0")).replace(".", "_")
        for key in mat_data.keys():
            if key == keyword:
                data = mat_data[key][:]
                try:
                    dfdata = pd.DataFrame(data)
                except ValueError as e:
                    print(e.message)
                    continue
        return dfdata

    def txt2dataframe(self):
        portfolios = []
        portfolios_price = []
        Nmonths = 0
        Nportfolios = 0
        if os.path.exists(self.path):
            with open(self.path, 'r') as f:
                for l in f:
                    line = l.rstrip('\n').rstrip().split('\t')[0]
                    portfolio = line.split('  ')
                    del (portfolio[0])
                    portfolio = np.array(portfolio)
                    portfolio = np.array(portfolio, dtype='float')
                    portfolios.append(portfolio)
                    if Nmonths == 0:
                        portfolios_price.append(portfolio)
                        Nportfolios = len(portfolio)
                    else:
                        price = portfolios_price[- 1] * portfolio
                        portfolios_price.append(price)
                    Nmonths += 1

        portfolios = np.array(portfolios)
        portfolios_price = np.array(portfolios_price)
        return Nportfolios, Nmonths, portfolios, portfolios_price



    def csv2dataframe(self, keyword):
        dfdata = pd.read_csv(self.path)
        init_time = str(dfdata.iloc[0][0])
        if "." in init_time:
            init_time = init_time.split(".")[0]
        portfolio = dfdata.values[:, 1:] * 0.01 + 1
        portfolio = portfolio.astype('float64')
        n, m = portfolio.shape
        if keyword == "portfolios":
            return portfolio
        elif keyword == "Nportfolios":
            return m
        elif keyword == "Nmonths":
            return n
        elif keyword == "portfolios_price":
            price = np.zeros(portfolio.shape)
            price[0] = portfolio[0]
            for i in range(1, n):
                price[i] = price[i - 1] * portfolio[i]
            return price
        elif keyword == "init_time":
            return init_time
        else:
            return None

    def get_stock_feature(self):
        feature_path = self.dict_path + "/stock_feature.csv"
        # feature_path = self.dict_path + "/stock_feature_pred.csv"
        dfdata = pd.read_csv(feature_path)
        stock_feature = dfdata.values[:, 3:]
        stock_feature = np.array(stock_feature, dtype='float')
        n, d = stock_feature.shape
        return stock_feature, d

    def get_market_feature(self):
        feature_path = self.dict_path + "/market_feature.csv"
        dfdata = pd.read_csv(feature_path)
        market_feature = dfdata.values[:, 2:]
        market_feature = np.array(market_feature, dtype='float')
        n, d = market_feature.shape
        return market_feature, d

    def random(self, choices):
        n = self.Nportfolios
        m = self.Nmonths
        self.Nportfolios = len(choices)
        not_choices = []
        for i in range(n - 1, -1, -1):
            if i not in choices:
                not_choices.append(i)
        not_choices = np.array(not_choices)
        self.portfolios = np.delete(self.portfolios, not_choices, 1)
        self.portfolios_price = np.delete(self.portfolios_price, not_choices, 1)
        for i in range(m - 1, -1, -1):
            self.stock_feature = np.delete(self.stock_feature, not_choices + i * n, 0)
        return self

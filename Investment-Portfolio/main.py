import copy
import datetime
import os

import pandas as pd

import strategy
from data_load.stocks import Stocks
from strategy.tools import *
from trade.portfolio import Portfolio
from trade.visulization import regret_plot
from trade.visulization import weight_plot
from trade.visulization import plot_cumulative_wealth

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
methods, methods_name = strategy.methods_config()
datasets, dataset_name = strategy.datasets_config()

cumulative_wealth = pd.DataFrame(columns=methods_name, index=dataset_name)
sharpe_ratio = pd.DataFrame(columns=methods_name, index=dataset_name)
volatility = pd.DataFrame(columns=methods_name, index=dataset_name)
max_drawdown = pd.DataFrame(columns=methods_name, index=dataset_name)
cumulative_wealth[cumulative_wealth != 0] = 0
sharpe_ratio[sharpe_ratio != 0] = 0
volatility[volatility != 0] = 0
max_drawdown[max_drawdown != 0] = 0
metrics = ['cumulative_wealth', 'sharpe_ratio', 'volatility', 'max_drawdown']

# 总实验次数，和选择股票个数(如果股票个数小于数据集中的股票个数，那么就是随机从所有股票中选择stock_size只股票来做组合）
# 大作业，可以重写tools.py中的getRandomN函数，来改变选股规则
duplicate = 1
stock_size = 36

for dataset in datasets:
    data_dict = PARENT_DIR + '/data/%s' % (dataset["name"])
    return_file = dataset["filename"]

    if "csv" in dataset["filename"]:
        stocks = Stocks(dict_path=data_dict, return_file=return_file, type="csv", is_context=True)
    elif "txt" in dataset["filename"]:
        stocks = Stocks(dict_path=data_dict, return_file=return_file, type="txt", is_context=True)
    else:
        stocks = Stocks(dict_path=data_dict, return_file=return_file)
    m = stocks.Nmonths
    n = stocks.Nportfolios

    if stocks.init_time is None:
        init_time = "1963-07-01"
    elif dataset["frequency"] == "month":
        init_time = datetime.datetime.strptime(stocks.init_time, '%Y%m')
        if init_time.month + dataset["init_t"] % 12 > 12:
            init_time = datetime.date(init_time.year + int(dataset["init_t"] / 12) + 1,
                                      init_time.month + (dataset["init_t"] % 12) - 12, init_time.day)
        else:
            init_time = datetime.date(init_time.year + int(dataset["init_t"] / 12),
                                      init_time.month + (dataset["init_t"] % 12), init_time.day)
    elif dataset["frequency"] == "day":
        init_time = datetime.datetime.strptime(stocks.init_time, '%Y/%m/%d')
        init_time = init_time + datetime.timedelta(days=dataset["init_t"])
    else:
        init_time = stocks.init_time

    print("dataset %s statistical:" % dataset["name"])
    print("init date = %s, total month = %d, portfolio number = %d, training date range = %s" % (
        init_time, m - dataset["span_t"] - dataset["init_t"], n, dataset["span_t"]))
    return_lists = pd.DataFrame()
    choices_lists = pd.DataFrame()
    net_return_list = pd.DataFrame()

    for i in range(duplicate):
        num_choice = stock_size
        choices = getRandomN(num_choice, n)
        choices_lists["dup"+str(i)] = choices
        stocks_random = copy.deepcopy(stocks)
        stocks_random.random(choices)
        portfolio = None
        all_method_time = pd.DataFrame()
        for idx, method in enumerate(methods):
            portfolio = Portfolio(stock=stocks_random, time_init=dataset["span_t"] + dataset["init_t"],
                                  frequency=dataset["frequency"])
            strategy.runPortfolio(stocks_random, portfolio, method, dataset)

            cumulative_wealth[method["name"]][dataset["name"]] += portfolio.eval(portfolio.cumulative_wealth)
            sharpe_ratio[method["name"]][dataset["name"]] += portfolio.eval(portfolio.sharpe_ratio)
            volatility[method["name"]][dataset["name"]] += portfolio.eval(portfolio.volatility)
            max_drawdown[method["name"]][dataset["name"]] += portfolio.eval(portfolio.max_drawdown)

            return_lists[method["name"]+"_dup"+str(i)] = portfolio.return_list
            net_return_list[method["name"] + "_dup" + str(i)] = portfolio.net_return_list

            cumulative_wealth.to_csv(PARENT_DIR + '/result/cumulative_wealth.csv', index=True, sep=',')
            sharpe_ratio.to_csv(PARENT_DIR + '/result/sharpe_ratio.csv', index=True, sep=',')
            volatility.to_csv(PARENT_DIR + '/result/volatility.csv', index=True, sep=',')
            max_drawdown.to_csv(PARENT_DIR + '/result/max_drawdown.csv', index=True, sep=',')
            return_lists.to_csv(PARENT_DIR + '/result/' + dataset["name"] + '.csv', index=False, sep=',')
            weight_plot(weight=portfolio.weight, title='weight-'+method["name"]+'-'+dataset["name"]+'-dup'+str(i), path=PARENT_DIR+'/result/')  # plot weight

        regret_plot(net_return_list=net_return_list, dup=i, methods_name=methods_name, title='regret-dup' + str(i), path=PARENT_DIR + '/result/')
        plot_cumulative_wealth(return_lists=return_lists, dataset=dataset["name"], frequency="month",
                               title='cumulative_wealth-dup' + str(i), path=PARENT_DIR + '/result/')
    net_return_list.to_csv(PARENT_DIR + '/result/' + dataset["name"] + '-net_return.csv', index=False, sep=',')
    return_lists.to_csv(PARENT_DIR + '/result/' + dataset["name"] + '.csv', index=False, sep=',')
    choices_lists.to_csv(PARENT_DIR + '/result/' + dataset["name"] + '_choice.csv', index=False, sep=',')

    for method in methods:
        cumulative_wealth[method["name"]][dataset["name"]] = cumulative_wealth[method["name"]][dataset["name"]] / duplicate
        sharpe_ratio[method["name"]][dataset["name"]] = sharpe_ratio[method["name"]][dataset["name"]] / duplicate
        volatility[method["name"]][dataset["name"]] = volatility[method["name"]][dataset["name"]] / duplicate
        max_drawdown[method["name"]][dataset["name"]] = max_drawdown[method["name"]][dataset["name"]] / duplicate

    cumulative_wealth.to_csv(PARENT_DIR + '/result/cumulative_wealth.csv', index=True, sep=',')
    sharpe_ratio.to_csv(PARENT_DIR + '/result/sharpe_ratio.csv', index=True, sep=',')
    volatility.to_csv(PARENT_DIR + '/result/volatility.csv', index=True, sep=',')
    max_drawdown.to_csv(PARENT_DIR + '/result/max_drawdown.csv', index=True, sep=',')

print("--------cumulative_wealth--------")
print(cumulative_wealth)
print("--------sharpe_ratio--------")
print(sharpe_ratio)
print("--------volatility--------")
print(volatility)
print("--------max_drawdown--------")
print(max_drawdown)

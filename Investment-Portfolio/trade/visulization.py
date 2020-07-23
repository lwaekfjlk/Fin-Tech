import matplotlib.pyplot as plt
import numpy as np

font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 23,
        }

colors = ["#00a1e9", "#71cae0", "#ffc000", "#058143", "#6bc233",  "#aecefa", "#e0996e",
          "#b9001f", "#0f3da0"]


def plot_cumulative_wealth(return_lists, dataset, title, path, frequency="month"):
    colors = ["#00a1e9", "#71cae0", "#ffc000", "#058143", "#BA55D3", "#b9001f"]
    markers = ["x", "^", "|", "+", "1", "*"]
    i = 0
    span = 20
    methods = return_lists.columns.tolist()
    for method in methods:
        if method == "Best_dup0":
            continue
        if dataset == "sandp_csv" or dataset == "ETFs_csv":
            span = 50
        return_list = return_lists[method].tolist()
        return_list_dot = return_list[::span]
        portfolio_name = method.replace("_dup0", "")
        plt.plot(range(0, len(return_list_dot) * span, span), return_list_dot, label=portfolio_name,
                 color=colors[i], marker=markers[i], linewidth=1)
        i += 1
    plt.xlabel('Trading Rounds', font)
    plt.ylabel('Cumulative Wealth', font)
    plt.legend(loc='best')
    plt.savefig(path + title + '.png', format='png')
    # plt.show()


def weight_plot(weight, title, path):
    timeline = np.arange(len(weight))
    plt.stackplot(timeline, np.asarray(weight).T, colors=colors)
    plt.title(title)
    plt.savefig(path + title + '.png', format='png')
    # plt.show()
    plt.close()


def regret_plot(net_return_list, dup, methods_name, title, path):
    postfix = "_dup" + str(dup)
    best_return = np.asarray(net_return_list["Best" + postfix])
    for i, method_name in enumerate(methods_name):
        if method_name == "Best":
            continue
        regrets = []
        regret = 0
        method_return = np.asarray(net_return_list[method_name + postfix])
        for idx, best_return_T in enumerate(best_return):
            if idx == 0:
                continue
            regret += np.log(best_return[idx]) - np.log(method_return[idx])
            regrets.append(regret)
        plt.plot(regrets, label=method_name, linewidth=1, color=colors[i+1])
    plt.xlabel('T', fontdict=font)
    plt.ylabel('Regret', fontdict=font)
    plt.legend(prop=font)
    plt.savefig(path + title + '.png', format='png')
    # plt.show()
    plt.close()
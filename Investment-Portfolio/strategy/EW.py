# EW Portfolio
# Inputs: m, n,  τ（span_t）

# Output:
# The portfolio weight vectors ωk and the portfolio returns µk
import numpy as np
import os
from data_load.stocks import Stocks
from trade.portfolio import Portfolio


span_t = 120


def EW_weight_compute(n, context=None):
    w = np.ones(n)
    w = w / n
    return w


if __name__ == "__main__":
    print("this is EW Portfolio")


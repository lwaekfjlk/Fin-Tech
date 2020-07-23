# Best Portfolio
# Inputs: m, n,  τ（span_t）

# Output:
# The portfolio weight vectors ωk
import numpy as np
import os
from data_load.stocks import Stocks
from trade.portfolio import Portfolio


span_t = 120


def Best_weight_compute(n, context):
    R = context["Rk"]

    w = np.zeros(n)
    w[np.argmax(R)] = 1
    return w


if __name__ == "__main__":
    print("this is Best Portfolio")

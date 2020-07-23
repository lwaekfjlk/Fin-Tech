# EG Portfolio
# Inputs: m, n,  τ（span_t）

# Output:
# The portfolio weight vectors ωk and the portfolio returns µk
import numpy as np
import os
from data_load.stocks import Stocks
from trade.portfolio import Portfolio
from cvxopt import blas, solvers
import cvxopt as opt

span_t = 120


def EG_weight_compute(n,context=None):
    # learning rate
    eta = 0.05
    # last iteration w
    w = context["wk-1"]
    # latest known reward
    R = context["Rk-1"]
    # calculate w
    w_x = np.dot(w, R)
    w = w*np.exp(eta*R/w_x)/ (np.sum(w*np.exp(eta*R/w_x)))
    return w

if __name__ == "__main__":
    print("this is MV Portfolio")


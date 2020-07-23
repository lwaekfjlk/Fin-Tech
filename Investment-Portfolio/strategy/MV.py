# MV Portfolio
# Inputs: m, n,  τ（span_t）

# Output:
# The portfolio weight vectors ωk and the portfolio returns µk
import numpy as np
import os
from data_load.stocks import Stocks
from trade.portfolio import Portfolio
from cvxopt import blas, solvers,matrix
import cvxopt as opt

span_t = 120


def MV_weight_compute(n, context=None):
    solvers.options['show_progress'] = False
    # use part of previous period time of data to make prediction
    # min w.T P w + Q w
    # Gx <= h
    # Ax = b
    P = opt.matrix(np.cov(context["R"],rowvar=False))
    Q = -0.001*opt.matrix(np.mean(context["R"],axis = 0).reshape(-1,1))
    G = opt.matrix(-np.eye(n))
    h = opt.matrix(0.0,(n,1))
    A = opt.matrix(np.ones((1,n)))
    b = opt.matrix(1.0)
    # optimize
    w = solvers.qp(P,Q,G,h,A,b)['x']
    w = np.array(w).reshape(1,-1)[0]
    return w


if __name__ == "__main__":
    print("this is MV Portfolio")


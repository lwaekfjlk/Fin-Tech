# ONS Portfolio
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


def ONS_weight_compute(n, context=None):
    # hyper parameter
    delta = 0.125
    beta = 1
    eta = 0
    # do iterated updating
    R = context["Rk-1"]
    w = context["wk-1"]
    # get gradient
    grad = np.mat(R / np.dot(w, R)).T
    # init A and b
    context["A"] += grad * grad.T
    context["b"] += (1 + 1./beta) * grad
    # do projection
    x = delta * context["A"].I * context["b"]
    M = context["A"]
    m = M.shape[0]
    P = opt.matrix(2 * M)
    Q = opt.matrix(-2 * M * x)
    G = opt.matrix(-np.eye(m))
    h = opt.matrix(np.zeros((m, 1)))
    A = opt.matrix(np.ones((1, m)))
    b = opt.matrix(1.)
    w = np.squeeze(solvers.qp(P, Q, G, h, A, b)['x'])

    return w * (1 - eta) + np.ones(n) / float(n) * eta


if __name__ == "__main__":
    print("this is ONS Portfolio")


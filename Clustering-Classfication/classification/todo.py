import numpy as np

# You may find below useful for Support Vector Machine
# More details in
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
#from scipy.optimize import minimize

def func_perceptron(X, y):
    '''
    Classification algorithm.

    Input:  X: Training sample features, P-by-N
            y: Training sample labels, 1-by-N

    Output: w: learned perceptron parameters, (P+1)-by-1
    '''
    P, N = X.shape
    w = np.zeros((P+1, 1))

    # generate matrix
    iter_num = 500
    X = np.row_stack((np.ones((1, N)),X))

    # epoch_size = iter_num
    for iter in range(iter_num):
        # batch_size = 1
        for i in range(N):
            x = X[:, i].reshape(P+1, 1)
            f_S = np.sign(np.dot(w.T, x))
            if f_S[0] != y[0][i]:
                w = w + x*y[0][i]

    return w



def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def func_logisticregression(X, y):
    '''
    Classification algorithm.

    Input:  X: Training sample features, P-by-N
            y: Training sample labels, 1-by-N

    Output: w: learned logistic regression parameters, (P+1)-by-1
    '''
    P, N = X.shape
    w = np.ones((P+1, 1))

    # generate matrix
    iter_num = 500
    alpha = 0.01
    X = np.row_stack((np.ones((1, N)),X))
    for iter in range(iter_num):
        for i in range(N):
            x_i = X[:, i].reshape(P+1, 1)
            sig = sigmoid(np.dot(x_i.T,w))
            if (y[0][i] == 1):
                error = 1 - sig
            else:
                error = 0 - sig
            w +=  alpha * x_i * error
    return w

def func_linearregression(X, y):
    '''
    Classification algorithm.

    Input:  X: Training sample features, P-by-N
            y: Training sample labels, 1-by-N

    Output: w: learned linear regression parameters, (P+1)-by-1
    '''
    P, N = X.shape
    w = np.ones((P+1, 1))

    # generate matrix
    iter_num = 500
    alpha = 0.01
    X = np.row_stack((np.ones((1, N)),X))
    for iter in range(iter_num):
        for i in range(N):
            x_i = X[:, i].reshape(P+1, 1)
            sig = np.dot(x_i.T,w)
            error = y[0][i] - sig
            w +=  alpha * x_i * error
    return w

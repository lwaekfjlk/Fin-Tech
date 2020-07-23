import numpy as np
from gen_data import gen_data
from plot import plot
from todo import func_perceptron
from todo import func_logisticregression
from todo import func_linearregression

no_iter = 100  # number of iteration
no_train = 70# YOUR CODE HERE # number of training data
no_test = 30# YOUR CODE HERE  # number of testing data
no_data = 100  # number of all data
assert(no_train + no_test == no_data)

cumulative_train_err = 0
cumulative_test_err = 0

for i in range(no_iter):
    X, y, w_f = gen_data(no_data)
    X_train, X_test = X[:, :no_train], X[:, no_train:]
    y_train, y_test = y[:, :no_train], y[:, no_train:]

    #w_g = func_linearregression(X_train, y_train)
    #w_g = func_logisticregression(X_train, y_train)
    w_g = func_perceptron(X_train, y_train)
    # Compute training, testing error
    # YOUR CODE HERE
    # ----------------
    # ANSWER BEGIN
    # ----------------
    # e.g
    # train_err = xxx
    # test_err = xxx
    train_label = np.ones((1,no_train))
    train_data = np.row_stack(( train_label,X_train))
    test_label = np.ones((1, no_test))
    test_data = np.row_stack((test_label,X_test))
    prediction_train = np.sign(np.dot(w_g.T, train_data))
    prediction_test = np.sign(np.dot(w_g.T, test_data))
    train_err = np.mean(np.abs(prediction_train-y_train))/2
    test_err = np.mean(np.abs(prediction_test-y_test))/2
    # ----------------
    # ANSWER END
    # ----------------
    cumulative_train_err += train_err
    cumulative_test_err += test_err

train_err = cumulative_train_err / no_iter
test_err = cumulative_test_err / no_iter

plot(X, y, w_f, w_g, "Classification")
print("Training error: %s" % train_err)
print("Testing error: %s" % test_err)
import numpy as np
from scipy.spatial.distance import cdist
MAXN = 100000

def kmeans(X, k):
    """
    K-Means clustering algorithm

    Input:  x: data point features, N-by-P maxtirx
            k: the number of clusters

    OUTPUT:  idx: cluster label, N-by-1 vector
    """
    N, P = X.shape
    idx = np.zeros(N)
    # YOUR CODE HERE
    # ----------------
    # ANSWER END
    # ----------------

    # init
    center = np.zeros((k, P))
    for i in range(k):
        center[i, :] = X[int(np.random.uniform(0, N)), :]

    # update
    while True:
        # index get
        for i in range(N):
            min_dist = MAXN
            for j in range(k):
                dist = np.sqrt(sum((X[i, :] - center[j, :])**2))
                if dist < min_dist:
                    min_dist = dist
                    idx[i] = j
        # check
        check_or_not = True
        for i in range(k):
            point_center = np.mean(X[np.nonzero(idx[:] == j)], axis=0)
            if not (point_center == idx[i]).all:
                check_or_not = False
                break
        if check_or_not == True:
            break

    # ----------------
    # ANSWER END
    # ----------------
    return idx



def spectral(W, k):
    """
    Spectral clustering algorithm

    Input:  W: Adjacency matrix, N-by-N matrix
            k: number of clusters

    Output:  idx: data point cluster labels, N-by-1 vector
    """
    N = W.shape[0]
    idx = np.zeros(N)
    # YOUR CODE HERE
    # ----------------
    # ANSWER BEGIN
    # ----------------

    # construct M
    D = np.sum(W, axis=1)
    L = np.diag(D) - W
    M = np.dot(np.diag(1.0/D), L)

    # select
    eigen, X = np.linalg.eig(M)
    # get
    min_eigen = np.sort(eigen)[1:k+1]
    pair = dict(zip(eigen, range(0, len(eigen))))

    index = [pair[k] for k in min_eigen]
    X = X[:, index]

    # ----------------
    # ANSWER END
    # ----------------
    X = X.astype(float)
    idx = kmeans(X, k)
    return idx


def knn_graph(X, k, threshold):
    '''
    Construct W using KNN graph

    Input:  X:data point features, N-by-P maxtirx.
            k: number of nearest neighbour.
            threshold: distance threshold.

    Output:  W - adjacency matrix, N-by-N matrix.
    '''
    N = X.shape[0]
    W = np.zeros((N, N))
    aj = cdist(X, X, 'euclidean')
    for i in range(N):
        index = np.argsort(aj[i])[:(k+1)]  # aj[i,i] = 0
        W[i, index] = 1
    W[aj >= threshold] = 0
    return W

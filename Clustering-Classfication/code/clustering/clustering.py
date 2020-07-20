import numpy as np
import scipy.io as sio
from plot import plot
from todo import kmeans
from todo import spectral
from todo import knn_graph

cluster_data = sio.loadmat('cluster_data.mat')
X = cluster_data['X']

idx = kmeans(X, 2)
plot(X, idx, "Clustering-kmeans")

W = knn_graph(X, 15, 1.45)  # recommend parameters
idx = spectral(W, 2)
plot(X, idx, "Clustering-Spectral")
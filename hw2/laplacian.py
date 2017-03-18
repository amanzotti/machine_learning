import numpy as np
import scipy


def distance_matrix(data):
    '''
    tested
    '''
    distances = np.zeros([data.shape[0], data.shape[0]])
    for (i, pointi) in enumerate(data):
        for (j, pointj) in enumerate(data):
            distances[i, j] = np.linalg.norm(pointi - pointj)
    return distances


def laplacian_eig(data, dim=2, k=10, sigma=0.5):

    dist = distance_matrix(data)

# build the graph
    graph = np.zeros(dist.shape)
    L2_dist = dist.copy()
# set k neigh connected disconnect the others
    for i in np.arange(dist.shape[0]):
        dist[i, i] = np.inf
        for j in np.arange(k):
            idx = dist[i].argmin()
            graph[i, idx] = 1.0
            graph[idx, i] = graph[i, idx]
            dist[i, idx] = np.inf
    print np.sum(dist[:,0]!=np.inf)
    # Step 2: Choosing the weights using the heat input

    nz = np.nonzero(graph)
    graph[nz] *= np.exp(-L2_dist[nz]**2 / sigma)

    # Laplacian matrix solve Lf=l Df
    weight = np.diag(graph.sum(1))
    # build laplacian l
    laplacian = weight - graph
    laplacian[np.isinf(laplacian)] = 0
    laplacian[np.isnan(laplacian)] = 0

    # Generalized Eigenvalue Decomposition
    # generalized problem
    val, vec = scipy.linalg.eig(laplacian, weight)
    index = np.real(val).argsort()
    # leave eigenvector 0 alone and return the firs d after that
    return vec[:, index[1:dim + 1]]

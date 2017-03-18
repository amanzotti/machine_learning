
# build cov matrix
import numpy as np
import matplotlib.pylab as plt

data = np.loadtxt('3Ddata.txt')
x = data[:, 0:3]


# build graph

# distance matrix
def distance_matrix(data):
    '''
    tested
    '''
    distances = np.zeros([data.shape[0], data.shape[0]])
    for (i, pointi) in enumerate(data):
        for (j, pointj) in enumerate(data):
            distances[i, j] = np.linalg.norm(pointi - pointj)
    return distances


def set_K_neighboorg(distances, k=10):
    '''
    tested
    '''
    output = np.ones_like(distances)
    # set not connected to inf
    output *= np.inf
    np.fill_diagonal(output, 0)

    for i in range(distances.shape[0]):
        closest = distances[:, i].argsort()[0:k]
        # print i, closest
        output[i, closest] = distances[i, closest]
        # output[closest, i] = distances[closest, i]
    return output


# apply Floyd algotrithm

def shortest_path(distances):
    output = distances.copy()
    # for i in range(output.shape[0]):
    #     output = np.minimum(output, np.add.outer(output[:, i], output[i, :]))

    # for k in range(output.shape[0]):
    #     for j in range(output.shape[0]):
    #         for j in range(output.shape[0]):
    #             output[i,j] = min(output[i,j], output[i,k] + output[k,j])
    for k in xrange(output.shape[0]):
        output = minimum(output, output[newaxis, k, :] + output[:, k, newaxis])

    # for k in xrange(output.shape[0]):
    #     for i in xrange(output.shape[0]):
    #         output[i,:] = minimum(output[i,:], output[i,k] + output[k,:])

    return output


# Below all give the same mds tested for debugging



# ========= I TRIED DIFFERENT MDS CAUSE IT DID NOT SEEM TO WORK AT FIRST
# do MDS on this
def mds_run(d, dimensions=2):
    '''
    not fully tested yet
    '''
    # create G_tilde from dist

    (n, n) = d.shape
    # create G_tilde − 1/2 P*D*P
    E = (-0.5 * d**2)

    # Use mat to get column and row means to act as column and row means.
    Er = np.mat(np.mean(E, 1))
    Es = np.mat(np.mean(E, 0))

    F = array(E - transpose(Er) - Es + mean(E))

    [U, S, V] = np.linalg.svd(F)

    # create data given the gram matrix using preposition 2

    Y = U * np.sqrt(S)

    return (Y[:, 0:dimensions], S)


# do MDS on this
def mds_run_2(data, dimensions=2):

    # create G_tilde from data

    data -= np.mean(data, axis=0)
    F = np.dot(data, data.T)
    [U, S, V] = np.linalg.svd(F)

    # create data given the gram matrix using preposition 2

    Y = U * np.sqrt(S)

    return (Y[:, 0:dimensions], S)


def cmdscale(D):

    # Number of points
    n = len(D)

    # Centering matrix
    H = np.eye(n) - np.ones((n, n)) / n

    # YY^T
    B = -H.dot(D**2).dot(H) / 2

    # Diagonalize
    evals, evecs = np.linalg.eigh(B)

    # Sort by eigenvalue in descending order
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Compute the coordinates using positive-eigenvalued components only
    w, = np.where(evals > 0)
    L = np.diag(np.sqrt(evals[w]))
    V = evecs[:, w]
    Y = V.dot(L)

    return Y, evals


def isomap(data, dim=2, k=10):
    # compute l2 distances among points
    dist = distance_matrix(data)
    # convert into a graph with k-neighbourhs
    #  deltaij = -inf if not connected
    dist_k = set_K_neighboorg(dist, k=k)
    # compute shortest path on the graph
    geodesic_distance = shortest_path(dist_k)
    # apply mds to it
    y, s = mds_run(geodesic_distance, dimensions=dim)
    return y

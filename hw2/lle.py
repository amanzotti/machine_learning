
# build cov matrix

import time
import numpy as np

import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

data = np.loadtxt('3Ddata.txt')
x = data[:, 0:3]

#! /usr/bin/env python
# -*- coding: utf-8 -*-


def distance_matrix(data):
    '''
    tested
    '''
    distances = np.zeros([data.shape[0], data.shape[0]])
    for (i, pointi) in enumerate(data):
        for (j, pointj) in enumerate(data):
            distances[i, j] = np.linalg.norm(pointi - pointj)
    return distances


def lle(data, dim=2, k=10):
    # compute L2 distance
    dist = distance_matrix(data)

    rn = xrange(dist.shape[0])
    dist[rn, rn] = np.inf

    neigh = dist.argsort(1)

    # STEP2: Solve for Reconstruction Weights
    # using a linear system is faster than inverting the matrix
    tol = 1e-3

    W = np.zeros((data.shape[0], k))
    for i in xrange(W.shape[0]):
        z = data[neigh[i, :k], :] - data[i]
        C = np.dot(z, z.T)
        # C_inv  = np.linalg.inv(C)
        # print np.sum(C_inv,axis=1)/np.sum(C_inv)
        # just a bit of tolerance in case it is singular
        C = C + np.eye(k) * tol * np.trace(C)
        W[i, :] = scipy.linalg.solve(C, np.ones((k, 1))).T
        W[i, :] /= W[i, :].sum()


    # Compute Embedding from Eigenvects of Cost Matrix M = (1 - W).T (1 - W)

    M = np.eye(data.shape[0])
    for i in xrange(M.shape[0]):
        w = W[i, :]
        j = neigh[i, :k]
        M[i, j] = M[i, j] - w
        M[j, i] = M[j, i] - w
        for l in xrange(w.shape[0]):
            M[j[l], j] = M[j[l], j] + w[l] * w

    # Calculation of Embedding
    val, vec = scipy.linalg.eig(M)
    index = np.real(val).argsort()
    index = index[::-1]

    return vec[:, index[-(dim + 1):-1]] * np.sqrt(data.shape[0])

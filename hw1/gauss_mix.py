import numpy as np
import matplotlib.pylab as plt
from scipy.stats import multivariate_normal as mvn
# initialize K
import kmeans
from numpy.core.umath_tests import matrix_multiply as mm


def initialize(X, k):
    pi = np.random.rand(k)
    pi /= np.sum(pi)
    mu = kmeans.initialize_kpp(X, k)
    sigmas = np.array([np.eye(2)] * k)
    return pi, mu, sigmas


def plot_models(xs, pis1, mus1, sigmas1):
    intervals = 101
    ys = np.linspace(-8, 8, intervals)
    X, Y = np.meshgrid(ys, ys)
    _ys = np.vstack([X.ravel(), Y.ravel()]).T
    ind = np.zeros([np.shape(xs)[0], 3])
    z = np.zeros(len(_ys))
    for i, (pi, mu, sigma) in enumerate(zip(pis1, mus1, sigmas1)):
        z += pi * mvn(mu, sigma).pdf(_ys)
        ind[:, i] = pi * mvn(mu, sigma).pdf(xs)

    indeces = np.argmax(ind, axis=1)
    z = z.reshape((intervals, intervals))

    # find indeces to make plot

    ax = plt.subplot(111)
    plt.scatter(mus1[:, 0], mus1[:, 1], alpha=1., c='r', marker='d', s=80)

    plt.scatter(xs[np.where(indeces == 0), 0], xs[np.where(indeces == 0), 1], alpha=0.5, c='b')
    plt.scatter(xs[np.where(indeces == 1), 0], xs[np.where(indeces == 1), 1], alpha=0.5, c='r')
    plt.scatter(xs[np.where(indeces == 2), 0], xs[np.where(indeces == 2), 1], alpha=0.5, c='k')

    plt.contour(X, Y, z, N=3)
    plt.axis([-6, 8, -6, 6])
    ax.axes.set_aspect('equal')
    plt.tight_layout()
    return 0


def MLE_gaussian_mix(data, weights, means, sigmas, tol=0.01, max_iter=100):

    n, p = data.shape
    k = len(weights)

    ll_old = 0
    for i in range(max_iter):

        like_new = 0

        # E-step
        ws = np.zeros((k, n))
        for j in range(k):
            ws[j, :] = weights[j] * mvn(means[j], sigmas[j]).pdf(data)
        ws /= ws.sum(0)

        # M-step
        weights = ws.sum(axis=1)
        weights /= n

        means = np.dot(ws, data)
        # vectorize this in python is damn hard you can do the trick with transpose
        # means /= ws.sum(1)[:, None]
        means = (means.T / ws.sum(1)).T

        sigmas = np.zeros((k, p, p))
        for j in range(k):
            ys = data - means[j, :]
            sigmas[j] = (ws[j, :, None, None] * mm(ys[:, :, None], ys[:, None, :])).sum(axis=0)
        sigmas /= ws.sum(axis=1)[:, None, None]

        # compute the  likelihoood anc compare
        like_new = 0
        for pi, mu, sigma in zip(weights, means, sigmas):
            like_new += pi * mvn(mu, sigma).pdf(data)
        like_new = np.log(like_new).sum()

        if np.abs(like_new - ll_old) < tol:
            break
        ll_old = like_new

    return like_new, weights, means, sigmas

# build cov matrix
import numpy as np
import matplotlib.pylab as plt

'''
tested
'''

# plot routines used in all the exercise


def plot_3d(x, ind):
    from mpl_toolkits.mplot3d import Axes3D
    colors = {1: 'g', 2: 'y', 3: 'b', 4: 'r'}

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in np.unique(ind):
        ax.scatter(x[np.where(ind == i)][:, 0], x[np.where(ind == i)][:, 1],
                   x[np.where(ind == i)][:, 2], c=colors[i], alpha=0.8)
    return 0


def plot_2d(x, ind):
    colors = {1: 'g', 2: 'y', 3: 'b', 4: 'r'}
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in np.unique(ind):
        ax.scatter(x[np.where(ind == i)][:, 0], x[np.where(ind == i)][:, 1], c=colors[i], alpha=0.8)
    return 0


def plot_1d(x, ind):
    colors = {1: 'g', 2: 'y', 3: 'b', 4: 'r'}
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in np.unique(ind):
        ax.plot(x[np.where(ind == i)][:, 0], c=colors[i], alpha=0.8)
    return 0

# pca starts here.

# # use
# data = np.loadtxt('3Ddata.txt')
# x = data[:, 0:3]
# pca(x)


def pca(x):
    # center
    x -= np.mean(x, axis=0)
    # compute conv
    sigma = np.dot(x.T, x) / x.shape[0]
    # eigenproblem
    values, vectors = np.linalg.eigh(sigma)
    # sort them
    idx = values.argsort()[::-1]
    # eigenValues = values[idx]
    eigenVectors = vectors[:, idx]

    # project into eigenvector and return
    return np.dot(x, eigenVectors[:, :2])

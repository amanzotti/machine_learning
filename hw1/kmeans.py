import numpy as np
import matplotlib.pylab as plt

# initialize K


def initialize(X, k):
    mu_i = np.random.randint(0, high=np.shape(X)[0], size=k)
    return X[mu_i, :]


def initialize_kpp(X, k):
    mu = np.zeros([k, np.shape(X)[1]])
    mu_i = np.random.randint(0, high=np.shape(X)[0])
    mu[0, :] = X[mu_i, :]
    for i in np.arange(1, k):
        D2 = np.array([min([np.linalg.norm(x - m)**2 for m in mu]) for x in X])
        probs = D2 / D2.sum()
        cumprobs = probs.cumsum()
        ind = np.where(cumprobs >= np.random.rand())[0][0]
        mu[i, :] = X[ind, :]
    return mu


def assigne(X, mu_list):
    distances = np.linalg.norm(X - mu_list[0], axis=1)**2

    for i, mu in enumerate(mu_list[1:]):
        dist = np.linalg.norm(X - mu, axis=1)**2
        distances = np.vstack([distances, dist])
    ind_assignment = np.argmin(distances, axis=0)

    return ind_assignment


def dist_functi(data, indeces, mu, k):
    cost_funct = 0
    for i in range(k):
        distances = np.sum(np.linalg.norm(data[np.where(indeces == i)] - mu[i], axis=1)**2)
        cost_funct += distances
    return cost_funct


def compute_new_mu(data, indeces, K):
    mu = np.zeros([K, np.shape(data)[1]])
    for i in np.arange(K):
        mu[i] = np.mean(data[np.where(indeces == i)], axis=0)
    return mu


def plot_res(data, ind, k):
    plt.figure()
    for i in np.arange(k):
        plt.scatter(data[np.where(ind[:, 0] == i)][:, 0], data[np.where(ind[:, 0] == i)][:, 1])


def run_kmeans(datafile='toydata.txt',  init='k++', k=3, iterate=20):

    data = np.loadtxt(datafile)
    k = 3
    cost = []

    ind_output = np.zeros([np.shape(data)[0], iterate])

    for i in np.arange(iterate):

        if init == 'k++':
            mu = initialize_kpp(data, k)
        else:
            mu = initialize(data, k)

        flag = 0
        cost_internal = []
        while flag == 0:
            ind = assigne(data, mu)
            cost_internal.append(dist_functi(data, ind, mu, k))
            new_mu = compute_new_mu(data, ind, k)
            if np.array_equal(new_mu, mu):
                flag = 1
            mu = new_mu
        ind_output[:, i] = ind
        cost.append([cost_internal])
    return cost, ind_output


def run_kmeans_bench(data,  init='k++', k=3, iterate=20):

    k = 3

    ind_output = np.zeros([np.shape(data)[0], iterate])
    for i in np.arange(iterate):

        if init == 'k++':
            mu = initialize_kpp(data, k)
        else:
            mu = initialize(data, k)
        flag = 0
        while flag == 0:
            ind = assigne(data, mu)
            new_mu = compute_new_mu(data, ind, k)
            if np.array_equal(new_mu, mu):
                flag = 1
            mu = new_mu
        ind_output[:, i] = ind
    return ind_output

if __name__ == "__main__":
    data = np.loadtxt('toydata.txt')
    plt.figure(1)
    plt.figure(2)
    k = 3
    mu = initialize_kpp(data, k)
    flag = 0
    cost = []
    while flag == 0:
        ind = assigne(data, mu)
        cost.append(dist_functi(data, ind, mu, k))
        new_mu = compute_new_mu(data, ind, k)
        print(new_mu)
        if np.array_equal(new_mu, mu):
            flag = 1
        mu = new_mu

    print('n interations', len(cost))

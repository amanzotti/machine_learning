import numpy as np
import matplotlib.pylab as plt
# KERNEL DEFINITION


def k_kernel(x, y):
    tau = 0.12
    return np.exp(-(x - y)**2 / (2. * tau))

# FIRST PART. DRAW FROM PRIOR

n_points = 100
z = np.linspace(0.01, 0.99, n_points)
K = np.zeros((n_points, n_points))
for i in range(n_points):
    for j in range(n_points):
        K[i, j] = k_kernel(z[i], z[j])

K += np.identity(n_points) * 0.0006
plt.clf()
for i in np.arange(20):
    f_z = np.zeros(n_points)
    f_z[0] = np.random.normal()
    # i=1

    K_ = K[0, 0]
    kxx_ = K[1, 1]
    kx = K[1, 0]
    K_inv = 1. / K_
    mean = np.dot(kx.T, np.dot(K_inv, f_z[:1].T))
    sigma = kxx_ - np.dot(kx.T, np.dot(K_inv, kx))
    f_z[1] = np.random.normal(mean, np.sqrt(sigma))

    for i in np.arange(2, n_points):
        K_ = K[:i, :i]
        kxx_ = K[i, i]
        kx = K[i, :i]
        K_inv = np.linalg.inv(K_)
        # mean = np.dot(kx.T, np.dot(K_inv, f_z[:i].T))
        mean = np.dot(kx.T, np.linalg.solve(K_, f_z[:i]))
        # sigma = kxx_ - np.dot(kx.T, np.dot(K_inv, kx))
        sigma = kxx_ - np.dot(kx.T, np.linalg.solve(K_, kx))

        # print i, sigma
        f_z[i] = np.random.normal(mean, np.sqrt(sigma))

    plt.plot(z, f_z, alpha=0.8)

plt.title('Sample from prior')
plt.savefig('sample_prior.pdf'.format(error_amp))

plt.clf()
for i in np.arange(20):
    mu = np.zeros(n_points)
    plt.plot(z, np.random.multivariate_normal(mu, K), alpha=0.6)
plt.title('Sample from prior')
plt.savefig('sample_prior_2.pdf'.format(error_amp))

plt.clf()

# ===============

# SECOND PART. WORK WITH DATA

data = np.loadtxt('gp.dat')
ind = np.argsort(data[:, 0])
data_sort = data[ind]
data_sort[:, 1] -= np.mean(data_sort[:, 1])


n_points = data_sort.shape[0]
K = np.zeros((n_points, n_points))
for i in range(n_points):
    for j in range(n_points):
        K[i, j] = k_kernel(data_sort[i, 0], data_sort[j, 0])

error_amp = 1.
K += np.identity(100) * error_amp
# K_inv = np.linalg.inv(K + np.identity(100) * 0.0001)
mu = np.linspace(0.01, 0.99, 100)
std = np.linspace(0.01, 0.99, 100)
z_mu = np.linspace(0.01, 0.99, 100)

for j, z in enumerate(z_mu):
    print j, z
    kx = np.ones(n_points)

    for i in np.arange(n_points):
        kx[i] = k_kernel(z, data_sort[i, 0])
    # std[j] = np.sqrt(1. - np.dot(kx.T, np.dot(K_inv, kx)))
    # mu[j] = np.dot(kx.T, np.dot(K_inv, y))
    std[j] = 2. * np.sqrt(1. - np.dot(kx.T, np.linalg.solve(K, kx)))
    mu[j] = np.dot(kx.T, np.linalg.solve(K, data_sort[:, 1]))
plt.clf()
plt.fill_between(z_mu, mu - std, mu + std, alpha=0.2, label='1 sigma range')
plt.plot(data_sort[:, 0], data_sort[:, 1], 'o', label='data')
plt.plot(z_mu, mu, '-o', label='mean')
plt.legend(loc=0)
plt.title('$\sigma = {}$'.format(error_amp))
plt.savefig('fig{}.pdf'.format(error_amp))
plt.clf()
# ===============

# Thid PART.Sample WITH DATA

n_points = 160
z = np.linspace(0.01, 0.99, n_points)

jointx = np.concatenate((data_sort[:, 0], z))


K = np.zeros((100 + n_points, 100 + n_points))
for i in range(n_points + 100):
    for j in range(n_points + 100):
        K[i, j] = k_kernel(jointx[i], jointx[j])


error_amp = 0.1
K += np.identity(100+n_points) * error_amp
plt.clf()

for i in np.arange(20):
    f_z = np.zeros(n_points)

    for i in np.arange(100, 100 + n_points):
        K_ = K[:i, :i]
        kxx_ = K[i, i]
        kx = K[i, :i]
        # mean = np.dot(kx.T, np.dot(K_inv, f_z[:i].T))
        mean = np.dot(kx.T, np.linalg.solve(K_, np.concatenate((data_sort[:, 1], f_z[:i - 100]))))
        # sigma = kxx_ - np.dot(kx.T, np.dot(K_inv, kx))
        sigma = kxx_ - np.dot(kx.T, np.linalg.solve(K_, kx))
        # print i, sigma
        f_z[i - 100] = np.random.normal(mean, np.sqrt(sigma))

    plt.plot(z, f_z, alpha=0.8)


plt.plot(data_sort[:, 0], data_sort[:, 1], 'o', label='data')
# plt.plot(z_mu, mu, '-o', label='mean')
plt.legend(loc=0)
plt.title('$\sigma = {}$, sample posterior'.format(error_amp))
plt.savefig('sample_posterior{}.pdf'.format(error_amp))
plt.clf()

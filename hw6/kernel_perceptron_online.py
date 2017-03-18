
import numpy as np
import time
import matplotlib.pylab as plt


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p


def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-np.linalg.norm(x - y)**2 / (2 * (sigma ** 2)))


kernel = linear_kernel

x_ = np.loadtxt('train2k.databw.35')
y = np.loadtxt('train2k.label.35')


# x_ = x_[:cut, :]
# y = y[:cut]

dim = x_.shape[1]
examples = x_.shape[0]

x = np.ones((examples, dim + 1))
x[:, 1:] = x_


alpha = np.zeros(examples)
# # compute the Gram matrix
# K = np.zeros((examples, examples))
# for i in range(examples):
#     for j in range(examples):
#         K[i, j] = kernel(x[i], x[j])


# np.save('kernel_linear_{}.npy'.format(examples), K)

# # np.save('kernel_linear_{}_sigma_{}.npy'.format(examples,sigma), K)


# K = np.load('kernel_gaussian_2000.npy')
K = np.load('kernel_linear_2000.npy')

errors = []
for i in range(examples):
    if np.sign(np.sum(K[:, i] * alpha * y)) != y[i]:
        errors.append(1)
        alpha[i] += 1.0
        # if n > 87 :
        #         plt.close()
        #         plt.figure()
        #         plt.matshow(x_[i].reshape((28, 28)), cmap='gray')
        #         plt.savefig('error_kenrel_{}_{}_whiletrue_{}.pdf'.format(i, y_tilde, y[i]))
        #         plt.close()
    else:
        errors.append(0)


# Support vectors
sv_mask = alpha > 1e-8
ind = np.arange(len(alpha))[sv_mask]
alpha = alpha[sv_mask]
sv = x[sv_mask]
sv_y = y[sv_mask]
print "%d support vectors out of %d points" % (len(alpha), examples)


errors_online = np.array(errors, dtype=np.int)
error_rate = np.zeros((examples, 2))
for i in xrange(1, examples - 1):
    error_rate[i, 0] = i / float(examples)
    error_rate[i, 1] = np.sum(errors[i:]) / float(len(errors[i:]))


plt.plot(error_rate[:, 0], error_rate[:, 1])
plt.xlabel('Proportion train')
plt.ylabel('Test Error Rate')
plt.savefig('kernel_linear_online.pdf')

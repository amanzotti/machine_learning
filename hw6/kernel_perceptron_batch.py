
import numpy as np
import time


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p


def gaussian_kernel(x, y, sigma=0.5):
    return np.exp(-np.linalg.norm(x - y)**2 / (2 * (sigma ** 2)))


kernel = gaussian_kernel

x_ = np.loadtxt('train2k.databw.35')
y = np.loadtxt('train2k.label.35')

cut = 2000

x_ = x_[:cut, :]
y = y[:cut]

dim = x_.shape[1]
examples = x_.shape[0]

x = np.ones((examples, dim + 1))
x[:, 1:] = x_


alpha = np.zeros(examples)
# compute the Gram matrix
K = np.zeros((examples, examples))
for i in range(examples):
    for j in range(examples):
        K[i, j] = kernel(x[i], x[j])


# np.save('kernel_linear_{}.npy'.format(examples), K)

# # np.save('kernel_linear_{}_sigma_{}.npy'.format(examples,sigma), K)


# K = np.load('kernel_gaussian_2000.npy')
# K = np.load('kernel_linear_2000.npy')

errors = []
for t in range(200):
    for i in range(examples):
        if np.sign(np.sum(K[:, i] * alpha * y)) != y[i]:
            errors.append(1)
            alpha[i] += 1.0
        else:
            errors.append(0)


# Support vectors
sv_mask = alpha > 1e-8
ind = np.arange(len(alpha))[sv_mask]
alpha = alpha[sv_mask]
sv = x[sv_mask]
sv_y = y[sv_mask]
print "%d support vectors out of %d points" % (len(alpha), examples)


test_ = np.loadtxt('test200.databw.35')
dim = test_.shape[1]
examples = test_.shape[0]
# initialize w
test = np.ones((examples, dim + 1))
test[:, 1:] = test_

label = []

for test_num in xrange(0, examples):
    s = 0
    for a, sv_y_loop, sv_loop in zip(alpha, sv_y, sv):
        s += a * sv_y_loop * kernel(test[test_num], sv_loop)
        # print s
    label.append(np.sign(s))


label = np.array(label, dtype=np.int)
np.savetxt('label_batch_kernel_gaussian.txt', label)
# np.savetxt('label_batch_kernel_linear.txt', label)


import numpy as np
from sklearn import cross_validation
import time

# def linear_kernel(x1, x2):
#     return np.dot(x1, x2)
# def polynomial_kernel(x, y, p=3):
#     return (1 + np.dot(x, y)) ** p


def gaussian_kernel(x, y, sigma):
    return np.exp(-np.linalg.norm(x - y)**2 / (2 * (sigma ** 2)))


re_run = [1,5,10,20,50,75,100,200]
re_run_test_error = []
x_ = np.loadtxt('train2k.databw.35')
y_ = np.loadtxt('train2k.label.35')\


for run in re_run:



    # print n
    # test_size = 600

    # x_ = x_[:-test_size, :]
    # y = y_[:-test_size]

    # dim = x_.shape[1]
    # examples = x_.shape[0]

    # x = np.ones((examples, dim + 1))
    # x[:, 1:] = x_

    dim = x_.shape[1]
    examples = x_.shape[0]

    x = np.ones((examples, dim + 1))
    x[:, 1:] = x_

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(x, y_, test_size=0.4, random_state=0)

    dim = X_train.shape[1]
    examples = X_train.shape[0]

    alpha = np.zeros(examples)
    # compute the Gram matrix
    K = np.zeros((examples, examples))
    for i in range(examples):
        for j in range(examples):
            K[i, j] = gaussian_kernel(X_train[i], X_train[j], sigma=1.0)


# np.save('kernel_linear_{}.npy'.format(examples), K)

    # np.save('kernel_linear_{}_sigma_{}.npy'.format(examples, sigma), K)


# K = np.load('kernel_gaussian_2000.npy')
# K = np.load('kernel_linear_2000.npy')

    errors = []
    for t in range(run):
        for i in range(examples):
            if np.sign(np.sum(K[:, i] * alpha * y_train)) != y_train[i]:
                errors.append(1)
                alpha[i] += 1.0
            else:
                errors.append(0)

    # Support vectors
    sv_mask = alpha > 1e-8
    ind = np.arange(len(alpha))[sv_mask]
    alpha = alpha[sv_mask]
    sv = X_train[sv_mask]
    sv_y = y_train[sv_mask]
    print "%d support vectors out of %d points" % (len(alpha), examples)

    # test_ = x_[-test_size:, :]
    # dim = test_.shape[1]
    # examples = test_.shape[0]
    # # initialize w
    # test = np.ones((examples, dim + 1))
    # test[:, 1:] = test_
    # y_test = y_[-test_size:]

    dim = X_test.shape[1]
    examples = X_test.shape[0]
    label = []

    for test_num in xrange(0, examples):
        s = 0.
        for a, sv_y_loop, sv_loop in zip(alpha, sv_y, sv):
            s += a * sv_y_loop * gaussian_kernel(X_test[test_num], sv_loop, sigma=1.0)
            # print s
        label.append(np.sign(s))

    re_run_test_error.append(np.mean(y_test == label))




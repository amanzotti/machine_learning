
import numpy as np
from sklearn import cross_validation


def linear_kernel(x1, x2):
    return np.dot(x1, x2)
# def polynomial_kernel(x, y, p=3):
#     return (1 + np.dot(x, y)) ** p


# def gaussian_kernel(x, y, sigma):
#     return np.exp(-np.linalg.norm(x - y)**2 / (2 * (sigma ** 2)))


re_run_test_error = []


x_ = np.loadtxt('train2k.databw.35')
y_ = np.loadtxt('train2k.label.35')


# test_size = 600

# x_ = x_[:-test_size, :]
# y = y_[:-test_size]
re_run_list = [1,5,10,20,50,75,100,200]

for re_run in re_run_list:
    dim = x_.shape[1]
    examples = x_.shape[0]

    x = np.ones((examples, dim + 1))
    x[:, 1:] = x_

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(x, y_, test_size=0.4, random_state=0)

    dim = X_train.shape[1]
    examples = X_train.shape[0]

    # compute the Gram matrix
    K = np.zeros((examples, examples))
    for i in range(examples):
        for j in range(examples):
            K[i, j] = linear_kernel(X_train[i], X_train[j])


    # np.save('kernel_linear_{}.npy'.format(examples), K)

    # np.save('kernel_linear_{}_sigma_{}.npy'.format(examples, sigma), K)

    w = x[0]
    errors = []
    # K = np.load('kernel_gaussian_2000.npy')
    # K = np.load('kernel_linear_2000.npy')
    for n in xrange(0, re_run):
        for i in xrange(1, examples):
            y_tilde = np.sign(np.dot(w, X_train[i]))
            if y_tilde * y_train[i] == 1:
                errors.append(0)
                continue
            else:
                errors.append(1)
                w += y_train[i] * X_train[i]


    # test_ = x_[-test_size:, :]
    dim = X_test.shape[1]
    examples = X_test.shape[0]
    # initialize w
    # test = np.ones((examples, dim + 1))
    # test[:, 1:] = test_
    # y_test = y_[-test_size:]
    label = []

    for test_num in xrange(0, examples):
        label.append(np.sign(np.dot(w, X_test[test_num])))

    re_run_test_error.append(np.mean(y_test == label))

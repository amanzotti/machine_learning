
import numpy as np
import matplotlib.pylab as plt

x_ = np.loadtxt('train2k.databw.35')
y = np.loadtxt('train2k.label.35')


dim = x_.shape[1]
examples = x_.shape[0]
# initialize w
x = np.ones((examples, dim + 1))
x[:, 1:] = x_

w = x[0]
errors = []
# eta = 0.2
# n = 100

# batch form
for i in xrange(1, examples):
    y_tilde = np.sign(np.dot(w, x[i]))
    if y_tilde * y[i] == 1:
        errors.append(0)
        continue
    else:
        errors.append(1)
        w += y[i] * x[i]

errors_online = np.array(errors, dtype=np.int)
error_rate = np.zeros((examples, 2))
for i in xrange(1, examples - 1):
    error_rate[i, 0] = i / float(examples)
    error_rate[i, 1] = np.sum(errors[i:]) / float(len(errors[i:]))


plt.plot(error_rate[:, 0], error_rate[:, 1])
plt.xlabel('Proportion train')
plt.ylabel('Test Error Rate')
plt.savefig('linear_online.pdf')

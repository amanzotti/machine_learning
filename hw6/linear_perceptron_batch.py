
import numpy as np
import matplotlib.pylab as plt
x_ = np.loadtxt('train2k.databw.35')
y = np.loadtxt('train2k.label.35')


dim = x_.shape[1]
examples = x_.shape[0]

for i in np.arange(examples):
    x_[i] /= np.linalg.norm(x_[i])

# initialize w
x = np.ones((examples, dim + 1))
x[:, 1:] = x_

w = x[0]
errors = []
# eta = 0.2
# n = 100

# batch form
for n in xrange(1, 200):
    for i in xrange(1, examples):
        y_tilde = np.sign(np.dot(w, x[i]))
        if y_tilde * y[i] == 1:
            errors.append(0)
            continue
        else:
            # if n > 87 :
            #     plt.close()
            #     plt.figure()
            #     plt.matshow(x_[i].reshape((28, 28)), cmap='gray')
            #     plt.savefig('error_{}_{}_whiletrue_{}.pdf'.format(i, y_tilde, y[i]))
            #     plt.close()

            errors.append(1)
            w += y[i] * x[i]

test_ = np.loadtxt('test200.databw.35')
dim = test_.shape[1]
examples = test_.shape[0]

for i in np.arange(examples):
    test_[i] /= np.linalg.norm(test_[i])


# initialize w
test = np.ones((examples, dim + 1))
test[:, 1:] = test_

label = []

for test_num in xrange(0, examples):
    label.append(np.sign(np.dot(w, test[test_num])))

label = np.array(label, dtype=np.int)
np.savetxt('label_batch_linear.txt', label)

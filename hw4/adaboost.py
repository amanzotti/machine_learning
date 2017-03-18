from __future__ import division
import numpy as np
import violajones
import time
import pickle
'''

'''


class AdaBoost:

    def __init__(self, row, col, Npos, Nneg, label):
        # training set should be (image,face or not)
        self.row = row
        self.col = col
        self.label = label
        self.Npos = Npos
        self.Nneg = Nneg
        self.features = violajones.getHaar("/project/kicp/manzotti/viola_jones/",
                                           self.row, self.col, self.Npos, self.Nneg)
        self.N = len(self.label)
        self.weights = np.ones((self.N, 1)) / self.N
        # store the index of each feature
        self.feature = []
        self.theta = []
        self.polarity = []
        self.predict = []
        self.train_error = []
        self.train_single_error = []
        self.false_positive = []
        self.false_negative = []

        self.threshold = 0.
        # store the weight of each features
        self.alpha = []
        self.modified_threshold = 0.

    def learn(self, test=False):

        '''
        This is one round of the adaboost classifier

        '''
        # evaluate the weak learner and get the error
        currentMin, theta, polarity, featureIdx, predict = violajones.getWeakClassifier(
            self.features, self.weights, self.label, self.Npos)
        # print 'in leanrn', len(predict)

        print currentMin, theta, polarity, featureIdx  # , predict
        self.polarity.append(polarity)
        errors = np.array(self.label != predict)

        predict[predict == 0] = -1.
        self.predict.append(predict[:, 0])
        self.theta.append(theta)
        # we want to weight the error.
        e = (errors * self.weights).sum()
        self.train_single_error.append(e)
        if test:
            return e
        # now get alpha
        alpha = 0.5 * np.log((1 - e) / e)
        print 'e=%.2f a=%.2f' % (e, alpha)
        w = np.zeros((self.N, 1))
        # reweights, wrong and right
        for i in range(self.N):
            if errors[i] == 1:
                w[i] = self.weights[i] * np.exp(alpha)
            else:
                w[i] = self.weights[i] * np.exp(-alpha)
        # normalize weight
        self.weights = w / w.sum()
        # append the feature used and its weight
        self.feature.append(featureIdx)
        self.alpha.append(alpha)

    def fit_image(self, image, row=64, col=64):
        '''
        pass a gray 64x64 image and predict
        '''
        # evaluate the weak learner and get the error
        intImg = np.zeros((row + 1, col + 1))
        intImg[1:row + 1, 1:col + 1] = np.cumsum(np.cumsum(image, axis=0), axis=1)

        features = violajones.computeFeature(intImg, row, col, numFeatures=17408)
        _temp_predict = []

        for weak_learner in np.arange(len(self.feature)):
            _predict = np.sign(self.polarity[weak_learner] *
                               (features[self.feature[weak_learner]] - self.theta[weak_learner]))
            _temp_predict.append(_predict)

        predict_full = np.sign(np.dot(np.array(self.alpha)[:, np.newaxis].T, _temp_predict) + self.threshold)
        return predict_full

    def evaluate(self):
        # evaluate using the learned rules
        predict_full = np.sign(np.dot(np.array(self.alpha)[:, np.newaxis].T, self.predict) + self.threshold)
        label = self.label.copy()
        label[label == 0] = -1.
        # print('test error = ', 1. - np.sum(predict_full == label[:, 0]) / self.N )
        return predict_full, 1. - np.sum(predict_full == label[:, 0]) / self.N
        # for img_idx in self.N:
        #     hx = [self.alpha[i] * self.features[self.feature](x) for i in range(NR)]
        #     print x, sign(l) == np.sign(sum(hx))

    def evaluate_2(self):
        '''
        # evaluate using the learned rules
        '''

        _temp_predict = []

        for weak_learner in np.arange(len(self.feature)):
            _predict = np.sign(self.polarity[weak_learner] *
                               (self.features[self.feature[weak_learner]] - self.theta[weak_learner]))
            _temp_predict.append(_predict)

            # print('test error = ', 1. - np.sum(predict_full == label[:, 0]) / self.N )

        predict_full = np.sign(np.dot(np.array(self.alpha)[:, np.newaxis].T, _temp_predict) + self.threshold)
        label = self.label.copy()
        label[label == 0] = -1.
        # now er have collect them let's combine them in the usual way.
        return predict_full, 1. - np.sum(predict_full == label[:, 0]) / self.N

    def evaluate_threshold(self, verbose=0):
        threshold = 0.0
        false_negative = 1.
        while (false_negative != 0):
            predict_full = np.sign(np.dot(np.array(self.alpha)[:, np.newaxis].T, self.predict) + threshold)
            if verbose > 0:
                print 'predict', predict_full
            threshold += 0.01
            false_negative = np.sum(predict_full[0, :self.Npos] == -1)
            if verbose > 0:
                print 'false', false_negative

        self.threshold = threshold
        if verbose > 0:
            print threshold
        # label = self.label.copy()
        # label[label == 0] = -1.
        # print('test error = ', 1. - np.sum(predict_full == label[:, 0]) / self.N )
        # return predict_full, 1. - np.sum(predict_full == label[:, 0]) / self.N

    def test_false_pos(self, iterations=12):
        false_pos = []
        for i in np.arange(iterations):
            self.learn()
            self.evaluate_threshold()
            predict, _temp = self.evaluate()
            false_pos.append(_temp * 2.)
        return false_pos

    def train(self, iter_times=10, tolerance=1e-2):
        for i in np.arange(iter_times):
            self.learn()
            pred, error = self.evaluate()
            self.train_error.append(error)
            if error < tolerance:
                return 0
        return 0


class Cascade_Ada:

    def __init__(self, row, col, Npos, Nneg, label, number_estimators):
        # training set should be (image,face or not)
        self.row = row
        self.col = col
        self.label = label
        self.Npos = Npos
        self.number_estimators = number_estimators
        self.ada_cascade = []
        self.pruned_negatives = set()
        self.current_layer = 0
        self.n_features_layer = np.linspace(4, 20, self.number_estimators).astype(int)
        self.false_positive = []
        self.false_negative = []
        self.train_time = []
        self.train_error = []
        for i in np.arange(number_estimators):
            self.ada_cascade.append(AdaBoost(row, col, Npos, Nneg, label))

    def run_train(self):
        for i in np.arange(self.number_estimators):
            tic = time.clock()
            print i, self.current_layer, list(self.pruned_negatives)
            # remove already negative for the next classifier in the cascade
            self.ada_cascade[self.current_layer].label = np.delete(
                self.ada_cascade[self.current_layer].label, list(self.pruned_negatives), axis=0)
            self.ada_cascade[self.current_layer].features = np.delete(
                self.ada_cascade[self.current_layer].features, list(self.pruned_negatives), axis=1)
            self.ada_cascade[self.current_layer].weights = np.delete(
                self.ada_cascade[self.current_layer].weights, list(self.pruned_negatives), axis=0)

            self.ada_cascade[self.current_layer].N = len(self.ada_cascade[self.current_layer].weights)
            # _N_pos = np.sum(label)
            # print 'shape before traineing',
            # self.ada_cascade[self.current_layer].label.shape,
            # self.ada_cascade[self.current_layer].weights.shape,
            # self.ada_cascade[self.current_layer].features.shape

            self.ada_cascade[self.current_layer].train(iter_times=self.n_features_layer[self.current_layer])
            self.ada_cascade[self.current_layer].evaluate_threshold()
            predict, error = self.ada_cascade[self.current_layer].evaluate()
            self.train_error.append(error)
            # get false positive and false negative
            self.false_positive.append(
                np.sum(np.all([[predict == 1], [(self.ada_cascade[self.current_layer].label == 0).T]], axis=0)[0][0]))

            print len(predict[0, :]), len(np.argwhere(predict == -1)[:, 1])
            self.pruned_negatives.update(tuple(np.argwhere(predict == -1)[:, 1]))
            toc = time.clock()
            self.train_time.append(toc - tic)
            # print len(list(self.pruned_negatives))
            self.current_layer += 1

    def sliding_window(self, test_image, patch_x_size=64, patch_y_size=64):
        # import time
        # import matplotlib.pylab as plt
        import multiprocessing
        test_size_y = np.shape(test_image)[0]
        test_size_x = np.shape(test_image)[1]
        faces = []
        for i in np.arange(100, test_size_y - patch_y_size, 8):
            for j in np.arange(100, test_size_x - patch_x_size, 8):
                # test_image[i:i + patch_y_size, j:j + patch_x_size]
                print i, j  # ,self.fit_image(test_image[i:i + patch_y_size, j:j + patch_x_size])
                if self.fit_image(test_image[i:i + patch_y_size, j:j + patch_x_size]) == 1:
                    faces.append([i, j])
                    print 'FOUND', i, j
                    # sys.exit()

        pickle.dump(faces, open("faces_found.p", "wb"))

        patch_list = [test_image[i:i + patch_x_size, j:j + patch_y_size]
                      for i in np.arange(100, test_size_y - patch_y_size, 8) for j in np.arange(100, test_size_x - patch_x_size, 8)]
        cpus = multiprocessing.cpu_count() - 2
        p = multiprocessing.Pool(cpus)
        detections = p.map(self.fit_image, patch_list)
        print detections
        return faces

    def no_overlap_plot(self, test_image, faces):
        import matplotlib.pylab as plt
        import scipy.ndimage
        from matplotlib.patches import Rectangle
        # turn faces list in array
        faces = np.array(faces)

        # use a gaussian smoothing and threeshold. This will exclude isolated point thare are probably not faces.
        A = np.zeros_like(test_image)
        A[faces[:, 0], faces[:, 1]] = 1
        new = np.zeros_like(test_image)
        new[scipy.ndimage.gaussian_filter(A, 10) > 0.005] = 1.
        boxes = np.zeros((len(np.argwhere(new == 1)[:, 0]), 4))
        boxes[:, 2] = np.argwhere(new == 1)[:, 0] + 64
        boxes[:, 3] = np.argwhere(new == 1)[:, 1] + 64
        boxes[:, 1] = np.argwhere(new == 1)[:, 1]
        boxes[:, 0] = np.argwhere(new == 1)[:, 0]
        new_faces = violajones.non_max_suppression_fast(boxes, 0.1)
        # simple no overlap use the first one with x in a -32 + 32
        #   now we are done simply plot.
        plt.imshow(test_image, cmap='gray')
        currentAxis = plt.gca()
        for i in np.arange(len(new_faces)):
            currentAxis.add_patch(
                Rectangle((new_faces[i, 1] - .1, new_faces[i, 0] - .1), 64, 64, fill=None, alpha=1, color='red', lw=1))

    def open_test_image(self, filepath='./class.jpg'):
        from scipy import misc
        test = misc.imread('./class.jpg', flatten=1)
        return test

    def fit_image(self, image, row=64, col=64):
        for stage in self.ada_cascade:
            _guess = stage.fit_image(image)
            if _guess == -1:
                return _guess
        return _guess

    # def scan_image(image):
    #     image
    #     pass

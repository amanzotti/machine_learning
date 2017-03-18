import numpy as np
from numpy import cumsum, matlib
import glob
from scipy import misc
import os
import matplotlib.pylab as plt
import sys

'''
Place this script in the folder containing
the faces/ and background/ directories or call
getHaar with the filepath containing the faces/ and background/
directories
'''

'''
Harr-like feature extraction for one image

Input
filepath: string, the name of the directory containing the faces/ and background/
directories
row, col: ints, dimensions of the training images
Npos: int, number of face images
Nneg: int, number of background images

Output
features: ndarray, extracted Haar features
'''


# def return_features_from_file(file, Npos, row, col, Nfeatures):

#     imgGray = misc.imread(file, flatten=1)  # array of floats, gray scale image
#     # convert to integral image
#     intImg = np.zeros((row + 1, col + 1))
#     intImg[1:row + 1, 1:col + 1] = np.cumsum(cumsum(imgGray, axis=0), axis=1)
#     # compute features
#     features[:, i] = computeFeature(intImg, row, col, Nfeatures)
#     return features


def get_haar_single_file(name_file):
    Nfeatures =61440# 245760  # change this number if you need to use more/less features
    imgGray = misc.imread(name_file, flatten=1)  # array of floats, gray scale image
    intImg = np.zeros((row + 1, col + 1))
    intImg[1:row + 1, 1:col + 1] = np.cumsum(cumsum(imgGray, axis=0), axis=1)
    # compute features
    return computeFeature(intImg, row, col, Nfeatures)


def getHaar(filepath, row, col, Npos, Nneg):
    Nimg = Npos + Nneg
    Nfeatures = 61440# 245760  # change this number if you need to use more/less features
    features = np.zeros((Nfeatures, Nimg))
    filename = filepath + 'features_' + 'n_pos_{}'.format(Npos) + '_n_neg_{}'.format(Nneg) + '.npy'
    print(filename)
    if os.path.exists(filename):
        print('loading file')
        features = np.load(filename)
        return features
    else:
        print('calcualting feature')

        files = glob.glob("faces/*.jpg")
        for i in xrange(len(files)):
            imgGray = misc.imread(files[i], flatten=1)  # array of floats, gray scale image
            if (i < Npos):
                # convert to integral image
                intImg = np.zeros((row + 1, col + 1))
                intImg[1:row + 1, 1:col + 1] = np.cumsum(np.cumsum(imgGray, axis=0), axis=1)
                # compute features
                features[:, i] = computeFeature(intImg, row, col, Nfeatures)

        files = glob.glob("background/*.jpg")
        for i in xrange(len(files)):
            imgGray = misc.imread(files[i], flatten=1)  # array of floats, gray scale image
            if (i < Nneg):
                # convert to integral image
                intImg = np.zeros((row + 1, col + 1))
                intImg[1:row + 1, 1:col + 1] = np.cumsum(cumsum(imgGray, axis=0), axis=1)
                #   print intImg.shape
                #   import pdb pdb.set_trace()
                # compute features
                features[:, i + Npos] = computeFeature(intImg, row, col, Nfeatures)

        np.save(filename, features.astype(np.float16))

        return features


'''
Given four corner points in the integral image
calculate the sum of pixels inside the rectangular.
'''


def sumRect(I, rect_four):

    row_start = rect_four[0]
    col_start = rect_four[1]
    width = rect_four[2]
    height = rect_four[3]
    one = I[row_start - 1, col_start - 1]
    two = I[row_start - 1, col_start + width - 1]
    three = I[row_start + height - 1, col_start - 1]
    four = I[row_start + height - 1, col_start + width - 1]
    rectsum = four + one - (two + three)
    if rectsum == np.inf:
        print('DEBUG', one, two, three, four, rectsum)
    return rectsum


'''
Computes the features. The cnt variable can be used to count the features.
If you'd like to have less or more features for debugging purposes, set the
Nfeatures =cnt in getHaar().
'''


def computeFeature(I, row, col, numFeatures):
    '''
    total row and col

    '''

    # print('computing feature')

    feature = np.zeros(numFeatures)

    # extract horizontal features
    cnt = 0  # count the number of features
    # This function calculates cnt=295937 features.

    window_h = 1
    window_w = 2  # window/feature size

    # h and w loop over all the possible shapes

    for h in xrange(3, row / window_h + 1, 2):  # extend the size of the rectangular feature
        for w in xrange(3, col / window_w + 1, 2):
            for i in xrange(1, row + 1 - h * window_h + 1, 4):  # stride size=4
                for j in xrange(1, col + 1 - w * window_w + 1, 4):
                    rect1 = np.array([i, j, w, h])  # 4x1
                    rect2 = np.array([i, j + w, w, h])
                    # rect1.append(np.array([i, j, w, h]))  # 4x1
                    # rect2.append(np.array([i, j + w, w, h]))
                    feature[cnt] = sumRect(I, rect2) - sumRect(I, rect1)
                    cnt = cnt + 1

    window_h = 2
    window_w = 1
    for h in xrange(3, row / window_h + 1, 2):
        for w in xrange(3, col / window_w + 1, 2):
            for i in xrange(1, row + 1 - h * window_h + 1, 4):
                for j in xrange(1, col + 1 - w * window_w + 1, 4):
                    rect1 = np.array([i, j, w, h])
                    rect2 = np.array([i + h, j, w, h])
                    feature[cnt] = sumRect(I, rect1) - sumRect(I, rect2)
                    cnt = cnt + 1

    print('cnt =', cnt)
    return feature


def visualizeFeature(row, col, index):

    feature_img = np.ones((row, col))
    # extract horizontal features
    cnt = 0  # count the number of features
    # This function calculates cnt=295937 features.

    window_h = 1
    window_w = 2  # window/feature size
    for h in xrange(3, row / window_h + 1, 2):  # extend the size of the rectangular feature
        for w in xrange(3, col / window_w + 1, 2):
            for i in xrange(1, row + 1 - h * window_h + 1, 4):  # stride size=4
                for j in xrange(1, col + 1 - w * window_w + 1, 4):
                    # rect1 = np.array([i, j, w, h])  # 4x1
                    # rect2 = np.array([i, j + w, w, h])
                    if cnt == index:
                        feature_img[i - 1:i + h - 1, j - 1: j + w - 1] *= 2.

                        feature_img[i - 1:i + h - 1, j + w - 1:j + w + w - 1] *= -2.
                        return feature_img

                    cnt = cnt + 1

    window_h = 2
    window_w = 1
    for h in xrange(3, row / window_h + 1, 2):
        for w in xrange(3, col / window_w + 1, 2):
            for i in xrange(1, row + 1 - h * window_h + 1, 4):
                for j in xrange(1, col + 1 - w * window_w + 1, 4):
                    # rect1 = np.array([i, j, w, h])
                    # rect2 = np.array([i + h, j, w, h])
                    if cnt == index:
                        feature_img[i - 1:i + h - 1, j - 1: j + w - 1] *= 2.

                        feature_img[i - 1:i + h - 1, j + w - 1:j + w + w - 1] *= -2.
                        return feature_img

                    cnt = cnt + 1
    print cnt

def print_face_feature(filepath, row, col, Npos, Nneg, feature_index):
    Nimg = Npos + Nneg
    feature_mask = visualizeFeature(row, col, feature_index)
    # filename = filepath + 'features_' + 'n_pos_{}'.format(Npos) + '_n_neg_{}'.format(Nneg) + '.npy'

    files = glob.glob(filepath + "faces/*.jpg")
    plt.figure()

    for i in xrange(len(files)):
        imgGray = misc.imread(files[i], flatten=1)  # array of floats, gray scale image
        if (i < Npos):
            # convert to integral image
            amin = np.amin(imgGray)
            amax = np.amax(imgGray)
            imgGray *= feature_mask
            plt.imshow(imgGray, interpolation='none', cmap=plt.cm.Greys, vmin=amin, vmax=amax)
            plt.savefig('./test{}.pdf'.format(i))
            plt.clf()

    files = glob.glob(filepath + "background/*.jpg")
    for i in xrange(len(files)):
        imgGray = misc.imread(files[i], flatten=1)  # array of floats, gray scale image
        if (i < Nneg):

            imgGray *= feature_mask

    return 0


'''
Select best weak classifier for each feature over all images

Input
features: ndarray, contains the features
weight: ndarray, vector of weights
label: ndarray, vector of labels
Npos: number of face images

Output:
currentMin: min weighted error
theta: threshold
polarity: polarity
featureIdx:  best feature index
bestResult: classification result. Note that this is equivalent
to h_t(x) from the original Viola-Jones paper and is used to determine the
strong classifier decision value by cascading several weak classifiers

'''

def eval_weak_class(i,features,tPos,tNeg):

       # get one feature for all images
    oneFeature = features[i, :]

    # sort feature to thresh for postive and negative
    sortedFeature = np.sort(oneFeature)
    sortedIdx = np.argsort(oneFeature)

    # sort weights and labels
    sortedWeight = weight[sortedIdx]
    sortedLabel = label[sortedIdx]

    # compute the weighted errors
    sPos = cumsum(np.multiply(sortedWeight, sortedLabel))
    sNeg = cumsum(sortedWeight) - sPos

    sPos = sPos.reshape(sPos.shape[0], 1)
    sNeg = sNeg.reshape(sNeg.shape[0], 1)
    errPos = sPos + (tNeg - sNeg)
    errNeg = sNeg + (tPos - sPos)

    # choose the threshold with the smallest error
    allErrMin = np.minimum(errPos, errNeg)  # pointwise min

    errMin = np.min(allErrMin)
    idxMin = np.argmin(allErrMin)

    # classification result under best threshold
    result = np.zeros((Nimgs, 1))
    if (errPos[idxMin] <= errNeg[idxMin]):
        p = 1
        end = result.shape[0]
        result[idxMin + 1:end] = 1
        result[sortedIdx] = np.copy(result)

    else:
        p = -1
        result[:idxMin + 1] = 1
        result[sortedIdx] = np.copy(result)

    # get the parameters that minimize the classification error
    # currentMin = errMin
    if (idxMin == 0):
        theta = sortedFeature[0] - 0.5
    elif (idxMin == Nfeatures - 1):
        theta = sortedFeature[Nfeatures - 1] + 0.5
    else:
        theta = (sortedFeature[idxMin] + sortedFeature[idxMin - 1]) / 2
    polarity = p
    featureIdx = i
    bestResult = result
    return polarity,featureIdx,bestResult

def getWeakClassifier(features, weight, label, Npos=None):

    # learn Npos if not passed
    if Npos is None:
        Npos = np.sum(label)

    Nfeatures, Nimgs = features.shape
    currentMin = np.inf
    # sum of the positive weight replicate for the number of positive images
    tPos = np.matlib.repmat(np.sum(weight[:Npos, 0]), Nimgs, 1)
    # same for negative
    tNeg = np.matlib.repmat(np.sum(weight[Npos:Nimgs, 0]), Nimgs, 1)

    for i in xrange(Nfeatures):
        # get one feature for all images
        oneFeature = features[i, :]

        # sort feature to thresh for postive and negative
        sortedFeature = np.sort(oneFeature)
        sortedIdx = np.argsort(oneFeature)

        # sort weights and labels
        sortedWeight = weight[sortedIdx]
        sortedLabel = label[sortedIdx]

        # compute the weighted errors
        sPos = cumsum(np.multiply(sortedWeight, sortedLabel))
        sNeg = cumsum(sortedWeight) - sPos

        sPos = sPos.reshape(sPos.shape[0], 1)
        sNeg = sNeg.reshape(sNeg.shape[0], 1)
        errPos = sPos + (tNeg - sNeg)
        errNeg = sNeg + (tPos - sPos)

        # choose the threshold with the smallest error
        allErrMin = np.minimum(errPos, errNeg)  # pointwise min

        errMin = np.min(allErrMin)
        idxMin = np.argmin(allErrMin)

        # classification result under best threshold
        result = np.zeros((Nimgs, 1))
        if (errPos[idxMin] <= errNeg[idxMin]):
            p = 1
            end = result.shape[0]
            result[idxMin + 1:end] = 1
            result[sortedIdx] = np.copy(result)

        else:
            p = -1
            result[:idxMin + 1] = 1
            result[sortedIdx] = np.copy(result)

        # get the parameters that minimize the classification error
        if (errMin < currentMin):
            currentMin = errMin
            if (idxMin == 0):
                theta = sortedFeature[0] - 0.5
            elif (idxMin == Nfeatures - 1):
                theta = sortedFeature[Nfeatures - 1] + 0.5
            else:
                theta = (sortedFeature[idxMin] + sortedFeature[idxMin - 1]) / 2
            polarity = p
            featureIdx = i
            bestResult = result

    return currentMin, theta, polarity, featureIdx, bestResult

# Malisiewicz et al.


def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


# Example usage:
if __name__ == "__main__":

    row = 64
    col = 64  # image size
    Npos = 2000  # number of face images
    Nneg = 2000  # number of background images

    features = getHaar("/project/kicp/manzotti/viola_jones/", row, col, Npos, Nneg)

    sys.exit()

    ''''
    # features = np.zeros((20,7))
    weight = np.zeros((7,1))
    weight [:,0] = [0.1250,0.1250,0.1250,0.1250,0.1667,0.1667,0.1667]
    label = np.zeros((7,1))
    print getWeakClassifier (features, weight, label, Npos)
    '''
    weight = np.zeros((4000, 1))
    weight[:, 0] = 1. / (Npos + Nneg)
    # weight [:,0] = [0.1250,0.1250,0.1250,0.1250,0.1667,0.1667,0.1667]
    label = np.zeros((4000, 1))
    label[:2000] = 1
    print getWeakClassifier(features, weight, label, Npos)

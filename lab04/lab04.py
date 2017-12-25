# USAGE
# python rbm.py -s 1
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import argparse
import time
import cv2
from struct import unpack
import gzip
from numpy import zeros, uint8, float32
import os
from PIL import Image


def get_labeled_data(image_file, label_file):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    # Open the images with gzip in read binary mode
    images = gzip.open(image_file, 'rb')
    labels = gzip.open(label_file, 'rb')

    # Read the binary data

    # We have to get big endian unsigned int. So we need '>I'

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)  # skip the magic_number
    n = labels.read(4)
    n = unpack('>I', n)[0]

    if number_of_images != n:
        raise Exception('number of labels did not match the number of images')

    # Get the data
    x = zeros((n, rows * cols), dtype=float32)  # Initialize numpy array
    y = zeros(n, dtype=uint8)  # Initialize numpy array
    for i in range(n):
        if i % 1000 == 0:
            print("i: %i" % i)
        for row in range(rows):
            for col in range(cols):
                tmp_pixel = images.read(1)  # Just a single byte
                tmp_pixel = unpack('>B', tmp_pixel)[0]
                x[i][row * rows + col] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]
    return x, y


def get_letters():
    pkg = 'letters/'
    images = os.listdir(pkg)
    labels, filtered = zip(*[(re.split('[-.]', image)[1], image) for image in images if image.endswith('.bmp')])
    opened_gray = [np.asarray(Image.open(pkg + image).convert('L').getdata()) for image in filtered]
    print(labels)
    return opened_gray, labels


def scale(X, eps=0.001):
    # scale the data points s.t the columns of the feature space
    # (i.e the predictors) are within the range [0, 1]
    return (X - np.min(X, axis=0)) / (np.max(X, axis=0) + eps)


def nudge(x, y):
    # initialize the translations to shift the image one pixel
    # up, down, left, and right, then initialize the new data
    # matrix and targets
    translations = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    data = []
    target = []

    # loop over each of the digits
    for (image, label) in zip(x, y):
        size = np.sqrt(len(image)).astype(int)
        # reshape the image from a feature vector of 784 raw
        # pixel intensities to a 28x28 'image'
        image = image.reshape(size, size)

        # loop over the translations
        for (tX, tY) in translations:
            # translate the image
            m = np.float32([[1, 0, tX], [0, 1, tY]])
            trans = cv2.warpAffine(image, m, (size, size))

            # update the list of data and target
            data.append(trans.flatten())
            target.append(label)

    # return a tuple of the data matrix and targets
    return np.array(data), np.array(target)


def find_optimal(x_train, y_train):
    x_train = scale(x_train)

    print("SEARCHING LOGISTIC REGRESSION")
    params = {"C": [1.0, 10.0, 100.0]}
    start = time.time()
    gs = GridSearchCV(LogisticRegression(), params, n_jobs=-1, verbose=1)
    gs.fit(x_train, y_train)

    print("done in %0.3fs" % (time.time() - start))
    print("best score: %0.3f" % gs.best_score_)
    print("LOGISTIC REGRESSION PARAMETERS")
    best_params = gs.best_estimator_.get_params()

    for p in sorted(params.keys()):
        print("\t %s: %f" % (p, best_params[p]))

    rbm = BernoulliRBM()
    logistic = LogisticRegression()
    classifier = Pipeline([("rbm", rbm), ("logistic", logistic)])

    print("SEARCHING RBM + LOGISTIC REGRESSION")
    params = {
        "rbm__learning_rate": [0.1, 0.01, 0.001],
        "rbm__n_iter": [20, 40, 80],
        "rbm__n_components": [50, 100, 200],
        "logistic__C": [1.0, 10.0, 100.0]}

    start = time.time()
    gs = GridSearchCV(classifier, params, n_jobs=-1, verbose=1)
    gs.fit(x_train, y_train)

    print("\ndone in %0.3fs" % (time.time() - start))
    print("best score: %0.3f" % gs.best_score_)
    print("RBM + LOGISTIC REGRESSION PARAMETERS")
    best_params = gs.best_estimator_.get_params()

    for p in sorted(params.keys()):
        print("\t %s: %f" % (p, best_params[p]))


def try_solution(x_train, y_train, x_test, y_test):
    x_train = scale(x_train)
    x_test = scale(x_test)

    logistic = LogisticRegression(C=1.0)
    logistic.fit(x_train, y_train)
    print("LOGISTIC REGRESSION ON ORIGINAL DATASET")
    print(classification_report(y_test, logistic.predict(x_test)))

    rbm = BernoulliRBM(n_components=200, n_iter=40,
                       learning_rate=0.01, verbose=True)
    logistic = LogisticRegression(C=1.0)

    classifier = Pipeline([("rbm", rbm), ("logistic", logistic)])
    classifier.fit(x_train, y_train)
    print("RBM + LOGISTIC REGRESSION ON ORIGINAL DATASET")
    print(classification_report(y_test, classifier.predict(x_test)))

    print("RBM + LOGISTIC REGRESSION ON NUDGED DATASET")
    (x_test, y_test) = nudge(x_test, y_test)
    print(classification_report(y_test, classifier.predict(x_test)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--search", type=int, default=0,
                    help="whether or not a grid search should be performed")
    args = vars(ap.parse_args())

    digits_X_train, digits_Y_train = get_labeled_data("digits/train-images-idx3-ubyte.gz",
                                                      "digits/train-labels-idx1-ubyte.gz")
    digits_X_test, digits_Y_test = get_labeled_data("digits/t10k-images-idx3-ubyte.gz",
                                                    "digits/t10k-labels-idx1-ubyte.gz")

    letters_X, letters_Y = get_letters()
    letters_X_train, letters_X_test, letters_Y_train, letters_Y_test = train_test_split(letters_X, letters_Y,
                                                                                        test_size=0.2)

    if args["search"] == 1:
        find_optimal(digits_X_train, digits_Y_train)
        # find_optimal(letters_X_train, letters_Y_train)
    else:
        try_solution(digits_X_train, digits_Y_train, digits_X_test, digits_Y_test)
        # try_solution(letters_X_train, letters_Y_train, letters_X_test, letters_Y_test)

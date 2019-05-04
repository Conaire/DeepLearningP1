import matplotlib
from keras.preprocessing.image import ImageDataGenerator

from src import SmallVGGNet

matplotlib.use("Agg")

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os


def getdata(dataset):
    data = []
    labels = []

    # grab the image paths and randomly shuffle them
    # imagePaths = sorted(list(paths.list_images(args["dataset"])))
    imagePaths = sorted(list(paths.list_images(dataset)))

    # random.shuffle(imagePaths)

    # imagePaths = imagePaths[0:120]
    random.seed(42)
    random.shuffle(imagePaths)
    imagePaths = imagePaths[0:2000]

    print(len(imagePaths))


    i = 0

    # loop over the input images
    for imagePath in imagePaths:
        # load the image, resize it to 64x64 pixels (the required input
        # spatial dimensions of SmallVGGNet), and store the image in the
        # data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (32, 32))
        data.append(image)

        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-3]
        labels.append(label)

        if i % 1000 == 0:
            print("so far," + str(i) + " files read ")

        i += 1

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    return data, labels

# import the necessary packages
import tflearn as tflearn
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os

from datetime import datetime
from scipy.io import loadmat
from sklearn.metrics import accuracy_score

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical
from tflearn.data_utils import samplewise_zero_center

class HwAlexNet:
    @staticmethod
    def build(width, height, depth, classes):


        activation = 'tanh'

        model = Sequential()

        # 1st Convolutional Layer
        model.add(Conv2D(filters=48, input_shape=(32, 32, 3), kernel_size=(8, 8), \
                         strides=(3, 3), padding='valid'))
        model.add(Activation(activation))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
        # Batch Normalisation before passing it to the next layer
        model.add(BatchNormalization())

        # 2nd Convolutional Layer
        model.add(Conv2D(filters=126, kernel_size=(2, 2), strides=(1, 1), padding='valid'))
        model.add(Activation(activation))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 3rd Convolutional Layer
        model.add(Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation(activation))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 4th Convolutional Layer
        model.add(Conv2D(filters=192, kernel_size=(2, 2), strides=(1, 1), padding='valid'))
        model.add(Activation(activation))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 5th Convolutional Layer
        model.add(Conv2D(filters=128, kernel_size=(2, 2), strides=(1, 1), padding='valid'))
        model.add(Activation(activation))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # Passing it to a dense layer
        model.add(Flatten())
        # 1st Dense Layer
        model.add(Dense(1024))     #, input_shape=(224 * 224 * 3,)
        model.add(Activation(activation))
        # Add Dropout to prevent overfitting
        model.add(Dropout(0.25))
        # Batch Normalisation
        ##aa model.add(BatchNormalization())

        # 2nd Dense Layer
        model.add(Dense(1024))
        model.add(Activation(activation))
        # Add Dropout
        model.add(Dropout(0.25))
        # Batch Normalisation
        ##aa model.add(BatchNormalization())

        # 3rd Dense Layer
        #model.add(Dense(1000))
        #model.add(Activation(activation))
        # Add Dropout
        #model.add(Dropout(0.4))
        # Batch Normalisation
        ##aa model.add(BatchNormalization())

        # Output Layer
        model.add(Dense(10))
        model.add(Activation('softmax'))

        model.summary()

        # (4) Compile
       # model.compile(loss='categorical_crossentropy', optimizer='adam', \
         #             metrics=['accuracy'])

        return model





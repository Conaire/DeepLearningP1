import keras
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
import numpy as np
from IPython.display import Image
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import matplotlib
from keras.preprocessing.image import ImageDataGenerator

from src import SmallVGGNet
from src.data import getdata

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

from src.data import getdata



args = {
    "dataset": "../tiny-imagenet-200/train",
    "plot": "output/simple_nn_plot.png",
    "model": "output/simple_nn.model",
    "label_bin": "output/simple_nn_lb.pickle",
}

data, labels = getdata(args["dataset"])

(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=0.25, random_state=42)


lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(len(lb.classes_),activation='softmax')(x) #final layer with softmax activation


model=Model(inputs=base_model.input,outputs=preds)



for i,layer in enumerate(model.layers):
    print(i,layer.name)

for layer in model.layers:
    layer.trainable = False
# or if we want to set the first 20 layers of the network to be non-trainable
for layer in model.layers[:20]:
    layer.trainable = False
for layer in model.layers[20:]:
    layer.trainable = True



model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy



model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])


EPOCHS = 10

H = model.fit(trainX, trainY, #validation_data=(testX, testY),
  #  steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, batch_size=32)


c_mat, f1, acc, f1_macro = model.evaluate(testX, testY)

print(acc)
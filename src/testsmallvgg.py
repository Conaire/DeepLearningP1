# set the matplotlib backend so figures can be saved in the background
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


args = {
    "dataset": "../tiny-imagenet-200/train",
    "plot": "output/simple_nn_plot.png",
    "model": "output/simple_nn.model",
    "label_bin": "output/simple_nn_lb.pickle",
}

print("[INFO] loading images...")


# grab the image paths and randomly shuffle them
#imagePaths = sorted(list(paths.list_images(args["dataset"])))
#imagePaths = sorted(list(paths.list_images(args["dataset"])))
data, labels = getdata(args["dataset"])

(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors (for 2-class, binary
# classification you should use Keras' to_categorical function
# instead as the scikit-learn's LabelBinarizer will not return a
# vector)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)


print(labels)


# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")


# initialize our VGG-like Convolutional Neural Network
model = SmallVGGNet.SmallVGGNet.build(width=64, height=64, depth=3,
	classes=len(lb.classes_))

# initialize our initial learning rate, # of epochs to train for,
# and batch size
INIT_LR=0.01
EPOCHS = 10
BS = 32

# initialize the model and optimizer (you'll want to use
# binary_crossentropy for 2-class classification)
print("[INFO] training network...")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,  metrics=["accuracy"])

# train the network
#H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
#                        validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
#                       epochs=EPOCHS)


H = model.fit(trainX, trainY, #validation_data=(testX, testY),
  #  steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, batch_size=32)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            labels=sorted(lb.classes_)))
                            #target_names=lb.classes_))

# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])


print("[INFO] serializing network and label binarizer...")
model.save(args["model"])
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()
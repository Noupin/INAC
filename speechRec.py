"""
Classifies .wav files to a given label
"""
__author__ = "Noupin, W&B"

#Third Party Imports
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM
from keras.utils import to_categorical
import wandb
from wandb.keras import WandbCallback
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

#First Party Imports
from tunableVariables import Tunable
from constants import Constants
from preprocessing import Preprocessing


wandb.init(project="speechrec")
prepro = Preprocessing()

#Save data to array file first
if not os.listdir(Constants.dataPath) == Constants.folderNames:
      prepro.save_data_to_array()

#Loading train set and test set
X_train, X_test, y_train, y_test = prepro.get_train_test()


X_train = X_train.reshape(X_train.shape[0], Tunable.buckets, Tunable.maxLen, Tunable.channels)
X_test = X_test.reshape(X_test.shape[0], Tunable.buckets, Tunable.maxLen, Tunable.channels)

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

model = Sequential([
    Conv2D(Tunable.convLayers,
          (3, 3),
          input_shape=(Tunable.buckets, Tunable.maxLen, Tunable.channels),
          activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(32,
          (3, 3),
          activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Dropout(0.3),
    Flatten(),
    Dropout(0.3),
    Dense(Constants.numClasses, activation='softmax')
])

model.compile(loss="categorical_crossentropy",
                  optimizer="rmsprop",
                  metrics=['accuracy'])

model.fit(X_train, y_train_hot, epochs=Tunable.epochs, validation_data=(X_test, y_test_hot), callbacks=[tf.keras.callbacks.ReduceLROnPlateau(), WandbCallback(data_type="image", labels=Constants.folderNames)])

model.save(Constants.modelPath + f"speechModel{Tunable.epochs}.model")

#model = tf.keras.models.load_model(r"C:\Coding\Models\audioModels\speechModel100.model")
#model.summary()

check = 100

print("\n\nExpected: {expected}\nPredicted: {predicted}".format(
                                                     expected=Constants.folderNames[int(y_train[check])],
                                                     predicted=Constants.folderNames[np.argmax(model.predict(X_train)[check])]))
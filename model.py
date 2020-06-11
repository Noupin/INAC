#pylint: disable=C0103, C0301
"""
Classifies .wav files to a given label
"""
__author__ = "Noupin, W&B"

#Third Party Imports
import numpy as np
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback

#First Party Imports
from tunableVariables import Tunable
from constants import Constants


wandb.init(project="speechrec")

class Model():
    """
    Holds the functions to create and use the Model
    """
    def __init__(self, preproVars):
        """
        Initialize the variables for Model
        """
        self.preproVars = preproVars
        self.model = None

    def createModel(self):
        """
        Create and compile AI model to be used
        """
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(Tunable.tunableDict['convFilters'],
                                   tuple(Tunable.tunableDict['convFilterSize']),
                                   input_shape=(Tunable.tunableDict['buckets'], Tunable.tunableDict['maxLen'], Tunable.tunableDict['channels']),
                                   activation=tf.nn.leaky_relu),
            tf.keras.layers.MaxPooling2D(pool_size=tuple(Tunable.tunableDict['poolSize'])),

            tf.keras.layers.Conv2D(Tunable.tunableDict['convFilters'],
                                   tuple(Tunable.tunableDict['convFilterSize']),
                                   activation=tf.nn.leaky_relu),
            tf.keras.layers.MaxPooling2D(pool_size=tuple(Tunable.tunableDict['poolSize'])),

            tf.keras.layers.Dropout(Tunable.tunableDict['dropoutVal']),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(Tunable.tunableDict['dropoutVal']),
            tf.keras.layers.Dense(Constants.numClasses, activation='softmax')
        ])

        self.model.compile(loss="categorical_crossentropy",
                           optimizer="rmsprop",
                           metrics=['accuracy'])

    def teachModel(self):
        """
        Trains the model over a given dataset
        """
        self.model.fit(self.preproVars.X_train, self.preproVars.y_train_hot,
                       epochs=Tunable.tunableDict['epochs'],
                       validation_data=(self.preproVars.X_test, self.preproVars.y_test_hot),
                       callbacks=[tf.keras.callbacks.ReduceLROnPlateau(),
                                  WandbCallback(data_type="image",
                                                labels=Constants.folderNames)])

    def saveModel(self):
        """
        Saves the model to diskspace
        """
        self.model.save(Constants.modelPath + f"speechModel{Tunable.tunableDict['epochs']}epochs{Tunable.tunableDict['BATCH_SIZE']}batch{Tunable.tunableDict['channels']}channels.model")

    def loadModel(self):
        """
        Loads a pretrained model to be used or trained more
        """
        self.model = tf.keras.models.load_model(Tunable.tunableDict['trainedModelPath'])

    def predict(self, index):
        """
        Uses a trained model to predict from a given dataset
        """
        print("\n\nExpected: {expected}\nPredicted: {predicted}".format(
            expected=Constants.folderNames[int(self.preproVars.y_train[index])],
            predicted=Constants.folderNames[np.argmax(self.model.predict(self.preproVars.X_train)[index])]))

#pylint: disable=C0103, C0301
"""
Classifies .wav files to a given label
"""
__author__ = "Noupin, W&B"

#Third Party Imports
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import wandb
from wandb.keras import WandbCallback

#First Party Imports
from tunableVariables import Tunable
from constants import Constants
from record import Record
import utilities


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
        self.recordSound = Record()

    def createModel(self):
        """
        Create and compile AI model to be used
        """
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(Tunable.tunableDict['convFilters'],
                                   tuple(Tunable.tunableDict['largeConvFilterSize']),
                                   input_shape=(Tunable.tunableDict['buckets'], Tunable.tunableDict['maxLen'], Tunable.tunableDict['channels']),
                                   activation=tf.nn.leaky_relu,
                                   padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=tuple(Tunable.tunableDict['poolSize'])),

            tf.keras.layers.Conv2D(Tunable.tunableDict['convFilters'],
                                   tuple(Tunable.tunableDict['mediumConvFilterSize']),
                                   activation=tf.nn.leaky_relu,
                                   padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=tuple(Tunable.tunableDict['poolSize'])),

            tf.keras.layers.Conv2D(Tunable.tunableDict['convFilters'],
                                   tuple(Tunable.tunableDict['mediumConvFilterSize']),
                                   activation=tf.nn.leaky_relu,
                                   padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=tuple(Tunable.tunableDict['poolSize'])),

            tf.keras.layers.Conv2D(Tunable.tunableDict['convFilters'],
                                   tuple(Tunable.tunableDict['smallConvFilterSize']),
                                   activation=tf.nn.leaky_relu,
                                   padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=tuple(Tunable.tunableDict['poolSize'])),

            tf.keras.layers.Dropout(Tunable.tunableDict['dropoutVal']),
            tf.keras.layers.Flatten(),

            tf.keras.layers.Dropout(Tunable.tunableDict['dropoutVal']),
            tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu),

            tf.keras.layers.Dropout(Tunable.tunableDict['dropoutVal']),
            tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu),

            tf.keras.layers.Dropout(Tunable.tunableDict['dropoutVal']),
            tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu),

            tf.keras.layers.Dropout(Tunable.tunableDict['dropoutVal']),
            tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu),

            tf.keras.layers.Dropout(Tunable.tunableDict['dropoutVal']),
            tf.keras.layers.Dense(Constants.numClasses, activation='softmax')
        ])

        self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                           optimizer=tf.keras.optimizers.RMSprop(0.001),
                           metrics=['accuracy'])
        self.model.summary()

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

    def predict(self, typeOf, index=None):
        """
        Uses a trained model to predict from a given dataset given an index and typeOf sound
        """
        if typeOf == "o":
            utilities.playWav(self.preproVars.X_train_sound[index][0])
            print("\n\nPredicted: {predicted}".format(
            predicted=Constants.folderNames[np.argmax(self.model.predict(self.preproVars.X_train)[index])]))
        elif typeOf == "n":
            self.recordSound.record()
            self.recordSound.save()
            audio = np.reshape(utilities.wav2mfcc(Constants.sampleWavFilePath), (1, 20, 20, 1))
            print("\n\nPredicted: {predicted}".format(
            predicted=Constants.folderNames[np.argmax(self.model.predict(audio))]))

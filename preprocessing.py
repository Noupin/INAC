#pylint: disable=C0103, C0301
"""
The functions needed to preprocess the data and make
the data usable for AI training
"""
__author__ = "Noupin, W&B"

#Third Party Imports
import os
import librosa
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import tensorflow as tf

#First Party Imports
from constants import Constants
from tunableVariables import Tunable
import utilities

class Preprocessing():
    """
    Hold the preprocessing functions and variabels that need preprocessed
    """

    def __init__(self):
        """
        Initializing variabels to be used across the class.
        """
        self.labels = os.listdir(Constants.dataPath)

        #Save data to array file first
        if not set(Constants.numpyNames).issubset(set(os.listdir(Constants.savePath))):
            self.save_data_to_array()
            self.save_filename_to_array()

        #Loading train set and test set
        self.X_train, self.X_test, self.y_train, self.y_test = self.get_train_test()
        self.X_train_sound, self.X_test_sound, self.y_train_sound, self.y_test_sound = self.get_filename_train_test()

        self.X_train = self.X_train.reshape(self.X_train.shape[0], Tunable.tunableDict['buckets'], Tunable.tunableDict['maxLen'], Tunable.tunableDict['channels'])
        self.X_test = self.X_test.reshape(self.X_test.shape[0], Tunable.tunableDict['buckets'], Tunable.tunableDict['maxLen'], Tunable.tunableDict['channels'])

        self.y_train_hot = tf.keras.utils.to_categorical(self.y_train)
        self.y_test_hot = tf.keras.utils.to_categorical(self.y_test)

        self.datasetSize = int(len(self.X_train)/(Tunable.tunableDict["pitchShiftUpper"] - Tunable.tunableDict["pitchShiftLower"]))

    def get_labels(self):
        """
        Takes an input of a folder path and outputs a
        tuple (Label, Indices of the labels, one-hot encoded labels)
        """
        label_indices = np.arange(0, len(self.labels))
        return self.labels, label_indices, tf.keras.utils.to_categorical(label_indices)

    def save_data_to_array(self):
        """
        Saves the .wav data from folders into arrays
        """

        for label in self.labels:
            mfcc_vectors = []
            wavFiles = []

            labelPath = os.path.join(Constants.dataPath, label)
            saveFilePath = f"{os.path.join(Constants.savePath, label)}.npy"

            for wavFile in os.listdir(labelPath):
                wavFiles.append(os.path.join(labelPath, wavFile))

            for wavFile in tqdm(wavFiles, "Saving vectors of label - '{}'".format(label)):
                mfcc = utilities.wav2mfccDataAugmnetation(wavFile)
                for data in mfcc:
                    mfcc_vectors.append(data)

            np.save(saveFilePath, mfcc_vectors)
    
    def save_filename_to_array(self):
        """
        Saves the sound file names from folders into arrays
        """
        for label in self.labels:
            wavFiles = []

            labelPath = os.path.join(Constants.dataPath, label)
            saveFilePath = f"{os.path.join(Constants.savePath, label)}SoundFiles.npy"

            for wavFile in os.listdir(labelPath):
                wavFiles.append([os.path.join(labelPath, wavFile)])
            
            np.save(saveFilePath, wavFiles)

    def get_train_test(self):
        """
        Spliting the data into train and test data
        """

        # Load file
        X = np.load(Constants.savePath + self.labels[0] + '.npy')
        y = np.zeros(X.shape[0])
        # Append all of the dataset into one single array, same goes for y
        for i, label in enumerate(self.labels[1:]):
            x = np.load(Constants.savePath + label + '.npy')
            X = np.vstack((X, x))
            y = np.append(y, np.full(x.shape[0], fill_value=(i + 1)))

        assert X.shape[0] == len(y)

        return train_test_split(X, y, test_size=(1 - Tunable.tunableDict['datasetSplit']), random_state=Tunable.tunableDict['datasetRandomState'], shuffle=True)
    
    def get_filename_train_test(self):
        """
        Spliting the sound files into train and test data
        """

        # Load file
        X = np.load(Constants.savePath + self.labels[0] + 'SoundFiles.npy')
        y = np.zeros(X.shape[0])
        # Append all of the dataset into one single array, same goes for y
        for i, label in enumerate(self.labels[1:]):
            x = np.load(Constants.savePath + label + 'SoundFiles.npy')
            X = np.vstack((X, x))
            y = np.append(y, np.full(x.shape[0], fill_value=(i + 1)))

        assert X.shape[0] == len(y)

        return train_test_split(X, y, test_size=(1 - Tunable.tunableDict['datasetSplit']), random_state=Tunable.tunableDict['datasetRandomState'], shuffle=True)


    def prepare_dataset(self):
        """
        Prepares the dataset to to be used in AI learning
        """
        data = {}
        for label in self.labels:
            labelPath = os.path.join(Constants.dataPath, label)
            vectors = []

            data[label] = {}
            data[label]['path'] = [Constants.dataPath+"/" +  wavfile for wavfile in os.listdir(labelPath)]

            for wavfile in data[label]['path']:
                wave, _ = librosa.load(wavfile, mono=True, sr=None)
                # Downsampling
                #wave = wave[::3]
                mfcc = librosa.feature.mfcc(wave, sr=Tunable.tunableDict['samplingRate'])
                vectors.append(mfcc)

            data[label]['mfcc'] = vectors

        return data

    def load_dataset(self):
        """
        Load the dataset from saved .npy files
        """
        data = self.prepare_dataset()

        dataset = []

        for key in data:
            for mfcc in data[key]['mfcc']:
                dataset.append((key, mfcc))

        return dataset[:]

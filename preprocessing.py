import librosa
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm

from constants import Constants
from tunableVariables import Tunable

class Preprocessing():
    """
    Hold the preprocessing functions and variabels that need preprocessed
    """

    def __init__(self):
        """
        Initializing variabels to be used across the class.
        """
        self.labels = os.listdir(Constants.dataPath)

    def get_labels(self):
        """
        Takes an input of a folder path and outputs a 
        tuple (Label, Indices of the labels, one-hot encoded labels)
        """
        label_indices = np.arange(0, len(self.labels))
        return self.labels, label_indices, to_categorical(label_indices)

    def wav2mfcc(self, file_path):
        """
        Convert file from .wav to Mel-Frequency Cepstral Coefficients
        """
        #Load .wav to array
        wave, _ = librosa.load(file_path, mono=True, sr=None)
        wave = np.asfortranarray(wave[::3])

        #Convert to Mel-Frequency Cepstral Coefficients
        mfcc = librosa.feature.mfcc(wave, sr=16000, n_mfcc=Tunable.buckets)

        # If maximum length exceeds mfcc lengths then pad the remaining ones
        if (Tunable.maxLen > mfcc.shape[1]):
            pad_width = Tunable.maxLen - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

        # Else cutoff the remaining parts
        else:
            mfcc = mfcc[:, :Tunable.maxLen]
        
        return mfcc

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
                mfcc = self.wav2mfcc(wavFile)
                mfcc_vectors.append(mfcc)

            np.save(saveFilePath, mfcc_vectors)

    def get_train_test(self, split_ratio=0.6, random_state=42):
        """
        Spliting the data into train and test data
        """

        # Getting first arrays
        X = np.load(Constants.savePath + self.labels[0] + '.npy')
        y = np.zeros(X.shape[0])

        # Append all of the dataset into one single array, same goes for y
        for i, label in enumerate(self.labels[1:]):
            x = np.load(Constants.savePath + label + '.npy')
            X = np.vstack((X, x))
            y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

        assert X.shape[0] == len(y)

        return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)


    def prepare_dataset(self, path=Constants.dataPath):
        data = {}
        for label in self.labels:
            labelPath = os.path.join(Constants.dataPath, label)
            vectors = []

            data[label] = {}
            data[label]['path'] = [Constants.dataPath+"/" +  wavfile for wavfile in os.listdir(labelPath)]

            for wavfile in data[label]['path']:
                wave, _ = librosa.load(wavfile, mono=True, sr=None)
                # Downsampling
                wave = wave[::3]
                mfcc = librosa.feature.mfcc(wave, sr=16000)
                vectors.append(mfcc)

            data[label]['mfcc'] = vectors

        return data

    def load_dataset(self):
        """
        Load the dataset from saved .npy files
        """
        data = self.prepare_dataset(Constants.dataPath)

        dataset = []

        for key in data:
            for mfcc in data[key]['mfcc']:
                dataset.append((key, mfcc))

        return dataset[:]

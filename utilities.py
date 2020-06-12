"""
No-self-use- functions
"""
__author__ = "Noupin"

#First Party Imports
import librosa
import numpy as np
from playsound import playsound

#Third Party Imports
from tunableVariables import Tunable
from constants import Constants

def wav2mfcc(file_path):
    """
    Convert file from .wav to Mel-Frequency Cepstral Coefficients
    """
    #Load .wav to array
    wave, _ = librosa.load(file_path, mono=Constants.channelMap[Tunable.tunableDict['channels']], sr=Tunable.tunableDict['samplingRate'])
    wave = np.asfortranarray(wave)

    #Convert to Mel-Frequency Cepstral Coefficients
    mfcc = librosa.feature.mfcc(wave, sr=Tunable.tunableDict['samplingRate'], n_mfcc=Tunable.tunableDict['buckets'])

    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if Tunable.tunableDict['maxLen'] > mfcc.shape[1]:
        pad_width = Tunable.tunableDict['maxLen'] - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='minimum')

    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :Tunable.tunableDict['maxLen']]

    return [mfcc]

def wav2mfccDataAugmnetation(file_path):
    """
    Convert file from .wav to Mel-Frequency Cepstral Coefficients
    and augment the data to enable better learning
    """
    #Load .wav to array
    augmentArray =[]
    wave, _ = librosa.load(file_path, mono=Constants.channelMap[Tunable.tunableDict['channels']], sr=Tunable.tunableDict['samplingRate'])
    for i in range(Tunable.tunableDict['pitchShiftLower'], Tunable.tunableDict['pitchShiftUpper']):
        wave = librosa.effects.pitch_shift(wave, sr=Tunable.tunableDict['samplingRate'], n_steps=i)
        wave = np.asfortranarray(wave)

        #Convert to Mel-Frequency Cepstral Coefficients
        mfcc = librosa.feature.mfcc(wave, sr=Tunable.tunableDict['samplingRate'], n_mfcc=Tunable.tunableDict['buckets'])

        # If maximum length exceeds mfcc lengths then pad the remaining ones
        if Tunable.tunableDict['maxLen'] > mfcc.shape[1]:
            pad_width = Tunable.tunableDict['maxLen'] - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='minimum')

        # Else cutoff the remaining parts
        else:
            mfcc = mfcc[:, :Tunable.tunableDict['maxLen']]
        augmentArray.append(mfcc)

    return augmentArray

def mfcc2wav(mfcc):
    wav = librosa.feature.inverse.mfcc_to_audio(mfcc)
    librosa.output.write_wav(Constants.sampleWavFilePath, wav, sr=Tunable.tunableDict['samplingRate'])

def playWav(filePath):
    playsound(filePath)
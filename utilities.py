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

    return mfcc

def reshapeMfcc(mfcc):
    shapedMfcc = []

    for _ in range(Tunable.tunableDict['buckets']):
        shapedMfcc.append([])

    subArrCount = 0
    for arr in mfcc:
        for subArr in arr:
            for val in subArr:
                shapedMfcc[subArrCount].append(val)
        subArrCount += 1
    
    return shapedMfcc

def mfcc2wav(mfcc):
    wav = librosa.feature.inverse.mfcc_to_audio(mfcc)
    librosa.output.write_wav(Constants.sampleWavFilePath, wav, sr=Tunable.tunableDict['samplingRate'])

def playWav(filePath):
    playsound(filePath)
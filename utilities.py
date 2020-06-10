"""
No-self-use- functions
"""
__author__ = "Noupin"

#First Party Imports
import librosa
import numpy as np

#Third Party Imports
from tunableVariables import Tunable

def wav2mfcc(file_path):
    """
    Convert file from .wav to Mel-Frequency Cepstral Coefficients
    """
    #Load .wav to array
    wave, _ = librosa.load(file_path, mono=True, sr=None)
    wave = np.asfortranarray(wave[::3])

    #Convert to Mel-Frequency Cepstral Coefficients
    mfcc = librosa.feature.mfcc(wave, sr=16000, n_mfcc=Tunable.buckets)

    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if Tunable.maxLen > mfcc.shape[1]:
        pad_width = Tunable.maxLen - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :Tunable.maxLen]

    return mfcc

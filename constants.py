#pylint: disable=C0103, C0301, R0903
"""
Holds a data class with non changing variables
"""
__author__ = "Noupin"

#Third Party Imports
import os

class Constants:
    """
    Constant variables in INAC
    """

    modelPath = r"C:\Coding\Models\audioModels/"
    dataPath = r"C:\Datasets\Audio\Wav/"
    savePath = r"C:\Datasets\Audio/"
    sampleWavFilePath = r"C:\Coding\Python\ML\Audio\INAC\sample.wav"

    folderNames = os.listdir(dataPath)
    folderNames.sort()

    numpyNames = folderNames.copy()
    for fileName in range(len(numpyNames)):
        numpyNames[fileName] = numpyNames[fileName] + ".npy"

    numClasses = len(folderNames)

    channelMap = {1: True, 2: False}

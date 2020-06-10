#pylint: disable=C0103, C0301, R0903
"""
Holds a data class with non changing variables
"""
__author__ = "Noupin"

class Constants:
    """
    Constant variables in INAC
    """

    modelPath = r"C:\Coding\Models\audioModels\\"
    dataPath = r"C:\Datasets\Audio\Wav\\"
    savePath = r"C:\Datasets\Audio\\"

    folderNames = ["bed", "happy", "cat"]
    folderNames.sort()
    numClasses = len(folderNames)

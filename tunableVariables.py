#pylint: disable=C0103, C0301, R0903
"""
Holds a data class with tunable variables
"""
__author__ = "Noupin"


class Tunable:
    """
    Tunable variables for INAC
    """
    maxLen = 20
    buckets = 20
    channels = 1
    samplingRate = 16000

    epochs = 50
    BATCH_SIZE = 64

    dropoutVal = 0.4
    convFilters = 16
    convFilterSize = (3, 3)
    poolSize = (2, 2)
    LSTMRemember = 16
    embeddingSize = 100

    datasetSplit = 0.7
    datasetRandomState = 42

    trainedModelPath = r"C:\Coding\Models\audioModels\speechModel50epochs64batch1channels.model"

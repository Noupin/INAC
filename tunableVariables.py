#pylint: disable=C0103, C0301, R0903
"""
Holds a data class with tunable variables
"""
__author__ = "Noupin"


class Tunable:
    """
    Tunable variables for INAC
    """
    maxLen = 11
    buckets = 20
    channels = 1

    epochs = 50
    BATCH_SIZE = 100

    dropoutVal = 0.4
    convFilters = 32
    convFilterSize = (3, 3)
    poolSize = (2, 2)

    trainedModelPath = r"C:\Coding\Models\audioModels\speechModel50epochs100batch1channels.model"

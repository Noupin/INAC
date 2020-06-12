#pylint: disable=C0103, C0301
"""
Classifies .wav files to a given label
"""
__author__ = "Noupin"

#Third Party Imports
import random
import time

#First Party Imports
from preprocessing import Preprocessing
from model import Model
from record import Record
import utilities


class INAC():
    """
    Given a dataset of .wav files you can create, train, save, load and use an AI model.
    """
    def __init__(self):
        """
        Initializing the variables needed
        """
        self.preproVars = Preprocessing()
        self.Model = Model(self.preproVars)

    def main(self):
        """
        Implementing a UI
        """
        loadOrCreate = ''
        predictBool = ''
        newOrOld = ''

        while loadOrCreate not in ('c', 'l', 'tl'):
            loadOrCreate = input("Would you like to load, create or train a loaded a model(l/c/tl): ").lower()

        if loadOrCreate == 'l':
            predictBool = 'y'

        while predictBool not in ('y', 'n'):
            predictBool = input("Would you like to see predictions from the model at the end(y/n): ").lower()

        if loadOrCreate == 'c':
            self.Model.createModel()
            self.Model.teachModel()
            self.Model.saveModel()
        elif loadOrCreate == 'tl':
            self.Model.loadModel()
            self.Model.teachModel()
            self.Model.saveModel()
        else:
            self.Model.loadModel()

        while predictBool == "y":
            while newOrOld not in ('n', 'o'):
                newOrOld = input("Would you lke to record a new sound or use an old sound(n/o): ")
            predictIdx = random.randint(0, self.preproVars.datasetSize)
            self.Model.predict(newOrOld, index=predictIdx)
            predictBool = input("Would you like to see another prediction(y/any char): ").lower()
            newOrOld = ''


CAS = INAC()

CAS.main()

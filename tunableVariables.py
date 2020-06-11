#pylint: disable=C0103, C0301, R0903
"""
Holds a data class with tunable variables
"""
__author__ = "Noupin"

import json

class Tunable:
    """
    Tunable variables for INAC
    """
    
    with open(r"C:\Coding\Python\ML\Audio\INAC\tunable.json") as jsonFile:
        tunableDict = json.load(jsonFile)

"""
Records a sound file for the AI to recognize
"""

#Third Party Imports
from sys import byteorder
from array import array
from struct import pack
import pyaudio
import wave

#First Party Imports
from tunableVariables import Tunable
from constants import Constants

class Record:

    def __init__(self):
        self.pyAud = None
        self.frames = int(Tunable.tunableDict["samplingRate"] / Tunable.tunableDict["framesPerBuffer"] * Tunable.tunableDict["recordingSeconds"])
        self.audio = []
    
    def record(self):
        """
        Records
        """
        self.pyAud = pyaudio.PyAudio()
        stream = self.pyAud.open(format=pyaudio.paInt16,
                                 channels=Tunable.tunableDict["channels"],
                                 rate=Tunable.tunableDict["samplingRate"],
                                 input=True,
                                 frames_per_buffer=Tunable.tunableDict["framesPerBuffer"])

        print("Recording Started.")
        for _ in range(0, self.frames):
            data = stream.read(Tunable.tunableDict["framesPerBuffer"])
            self.audio.append(data)
        print("Recording Finished.")
        stream.stop_stream()
        stream.close()
        self.pyAud.terminate()

    def save(self):
        """
        Saves the recording to a wav file
        """
        wavFile = wave.open(Constants.sampleWavFilePath, 'wb')
        wavFile.setnchannels(Tunable.tunableDict["channels"])
        wavFile.setsampwidth(self.pyAud.get_sample_size(pyaudio.paInt16))
        wavFile.setframerate(Tunable.tunableDict["samplingRate"])
        wavFile.writeframes(b''.join(self.audio))
        del self.pyAud
        self.audio.clear()

"""
Converts an input byte sequence to a vector
"""
import numpy as np

class ChainerRNNVectorizer():

    def __init__(self, model_name):
        pass

    def vectorize(self,buffer):
        """
        Take the buffer and compute an embedding

        :param buffer: Sequence of bytes
        :return: numpy array containing embedding of buffer
        """
        return buffer
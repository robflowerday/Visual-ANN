import numpy as np


def ReLU(Z):
    """ Rectified Linear Unit Activation function.
        Can take an number, or vector as input,
        if the input is a vector, the function is
        applied element-wise. The function returns
        the input value if positive and 0 otherwise."""
    return np.maximum(0, Z)


def dReLU(Z):
    """ Differential of the Rectified Linear Unit
        activation function. """
    return Z > 0


def sigmoid(Z):
    """ Sigmoid activation function. Returns a
        value between 0 and 1. """
    return 1 / (1 + np.exp(-Z))


def dsigmoid(Z):
    """ Differential of the Sigmoid activation
        function. """
    return Z * (1.0 - Z)

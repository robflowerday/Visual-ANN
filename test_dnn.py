import gzip
import _pickle
from random import randint
import sys
import time

import numpy as np

from activation_functions import (
    dReLU,
    dsigmoid,
    ReLU,
    sigmoid,
)
from dnn import DNN


# Instantiate a dense neural dnn class object with the
# number of neurons in its input layer and its learning rate.
dnn = DNN(num_input_neurons=28*28, learning_rate=0.3)

# Add hidden layers, and an output layer to the dnn, stating
# the number of neurons, activation function to be used at each
# layer of the dnn and the differential of the activation
# function.
dnn.add_layer(
    num_neurons=2,
    activation_function=ReLU,
    diff_activation_function=dReLU,
)
dnn.add_layer(
    num_neurons=3,
    activation_function=ReLU,
    diff_activation_function=dReLU,
)
dnn.add_layer(
    num_neurons=3,
    activation_function=ReLU,
    diff_activation_function=dReLU,
)
dnn.add_layer(
    num_neurons=10,
    activation_function=sigmoid,
    diff_activation_function=dsigmoid,
)

# Tell the dnn to initialise parameter vectors and lists.
# This must be done after all the layers in the dnn have
# been defined.
dnn.initialise_parameter_arrays()

# Define the data that will train the dnn.

# Load the MNIST training data which has 60,000 labeled training
# samples and 10,000 labeled testing samples of 28 x 28 grayscale
# images of handwritten digits from 0 - 9.
# When loaded as below, train_X, train_y, test_X and test_y
# become numpy arrays of sizes: (60000, 28, 28), (60000,),
# (10000, 28, 28) and (10000,) respectively.

# from keras.datasets.mnist import load_data
# (train_X, train_y), (test_X, test_y) = load_data()

# could also be used, but means another large dependency and
# keras has a habit of throwing errors on installation.
f = gzip.open('data/mnist.pkl.gz', 'rb')
if sys.version_info < (3,):
    data = _pickle.load(f)
else:
    data = _pickle.load(f, encoding='bytes')
f.close()
(train_X, train_y), (test_X, test_Y) = data


def extract_samples(
        num_samples=100,
        train_X=train_X,
        train_y=train_y,
        num_classes=10,
):
    """ Return a specified number of samples with one-hot encoded labels
        from a dataset. (Only currently tested to work with the MNIST
        dataset.) """
    # Select only a portion of the dataset
    i = randint(0, train_y.shape[0]-num_samples)
    train_X = train_X[i : i + num_samples]
    train_y = train_y[i : i + num_samples]

    # Flatten the input array so that there is a 1-1 mapping from a 1D input array
    # to the dnn's input neurons.
    train_X = train_X.reshape(train_X.shape[0], train_X.shape[1] * train_X.shape[2])

    # One hot encode the train_y array for the stated number of output classes.
    train_y = np.eye(num_classes)[train_y]

    return train_X, train_y


def train_on_selection_of_mnist_samples(dnn, iterations, num_samples):
    """ Train the dnn using forward and back propagation.

        The calculation of this function is all performed in the
        dnn.train() method, the rest of the function deals with
        looping and logging. """
    for i in range(iterations):
        train_X, train_y = extract_samples(num_samples=num_samples)
        start_time = time.time()

        dnn.train(
            training_sample_inputs=train_X,
            training_sample_results=train_y,
        )

        end_time = time.time()
        print(f"{i}: Cost: {dnn.cost}   - took {end_time - start_time} seconds to run.")


train_on_selection_of_mnist_samples(
    dnn=dnn,
    iterations=100,
    num_samples=10,
)

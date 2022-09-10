import _pickle
import gzip
import sys

import numpy as np

from activation_functions import ReLU, sigmoid
from cnn import CNN

cnn = CNN(input_image_shape=(1, 28, 28))
cnn.add_conv_layer(activation_function=ReLU, num_kernels=2, kernel_width=3, kernel_height=3)
cnn.add_conv_layer(activation_function=ReLU, num_kernels=2, kernel_width=3, kernel_height=3)
cnn.add_conv_layer(activation_function=ReLU, num_kernels=2, kernel_width=3, kernel_height=3)
cnn.add_flatten_layer()
cnn.add_dense_layer(activation_function=sigmoid, num_neurons=10)

input_array = np.array([np.array([
    np.array([
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
        np.array([21, 22, 23, 24, 25, 26, 27, 28, 29, 30]),
        np.array([31, 32, 33, 34, 35, 36, 37, 38, 39, 40]),
        np.array([41, 42, 43, 44, 45, 46, 47, 48, 49, 50]),
        np.array([51, 52, 53, 54, 55, 56, 57, 58, 59, 60]),
        np.array([61, 62, 63, 64, 65, 66, 67, 68, 69, 70]),
        np.array([71, 72, 73, 74, 75, 76, 77, 78, 79, 80]),
        np.array([81, 82, 83, 84, 85, 86, 87, 88, 89, 90]),
        np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100]),
    ]),
    np.array([
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
        np.array([21, 22, 23, 24, 25, 26, 27, 28, 29, 30]),
        np.array([31, 32, 33, 34, 35, 36, 37, 38, 39, 40]),
        np.array([41, 42, 43, 44, 45, 46, 47, 48, 49, 50]),
        np.array([51, 52, 53, 54, 55, 56, 57, 58, 59, 60]),
        np.array([61, 62, 63, 64, 65, 66, 67, 68, 69, 70]),
        np.array([71, 72, 73, 74, 75, 76, 77, 78, 79, 80]),
        np.array([81, 82, 83, 84, 85, 86, 87, 88, 89, 90]),
        np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100]),
    ]),
    np.array([
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
        np.array([21, 22, 23, 24, 25, 26, 27, 28, 29, 30]),
        np.array([31, 32, 33, 34, 35, 36, 37, 38, 39, 40]),
        np.array([41, 42, 43, 44, 45, 46, 47, 48, 49, 50]),
        np.array([51, 52, 53, 54, 55, 56, 57, 58, 59, 60]),
        np.array([61, 62, 63, 64, 65, 66, 67, 68, 69, 70]),
        np.array([71, 72, 73, 74, 75, 76, 77, 78, 79, 80]),
        np.array([81, 82, 83, 84, 85, 86, 87, 88, 89, 90]),
        np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100]),
    ]),
])])

class_values = np.array([0, 0, 1])


f = gzip.open('data/mnist.pkl.gz', 'rb')
if sys.version_info < (3,):
    data = _pickle.load(f)
else:
    data = _pickle.load(f, encoding='bytes')
f.close()
(train_X, train_y), (test_X, test_Y) = data

train_X = train_X[:10]
# train_X.reshape(10, 1, 28, 28)
train_y = train_y[:10]

# cnn.forward_prop(input_array, class_values)
# cnn.forward_prop(train_X, train_y)
cnn.forward_and_back_prop(train_X, train_y)
# cnn.print_last_layer_activation_values()

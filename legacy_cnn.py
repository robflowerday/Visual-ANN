from typing import List

import numpy as np

from activation_functions import sigmoid


class CNN:
    """ A class to represent a convolutional neural network. """

    def __init__(self, input_image_shape: tuple):
        self.activation_functions: List = []

        self.weights = []
        self.biases = []
        self.activations = []

        self.num_conv_layers = 0
        self.num_dense_layers = 0

        self.neurons_structure = [input_image_shape]

    def add_conv_layer(
            self,
            num_kernels: int,
            kernel_size: int,
            activation_function: callable,
            padding_size: int = 0,
            stride: int = 1,
    ):
        """ Adds a layer to the network. """
        # weights = [
        #     [  # np.random.rand((num_output_channels, num_input_channels, kernel_width, kernel_height))
        #         [1, 2, 3],
        #         [4, 5, 6],
        #         [7, 8, 9],
        #     ],
        #     [
        #         [1, 2, 3],
        #         [4, 5, 6],
        #         [7, 8, 9],
        #     ],
        #     [
        #         [1, 2, 3],
        #         [4, 5, 6],
        #         [7, 8, 9],
        #     ],
        # ]
        # if self.num_conv_layers == 0:
        #     if len(self.neurons_structure[0]) < 3:
        #         self.weights.append(np.random.rand(1, num_kernels, kernel_size, kernel_size))
        #     else:
        #         self.weights.append(np.random.rand(self.neurons_structure[0][0], num_kernels, kernel_size, kernel_size))
        # else:
        #     self.weights.append(np.random.rand(self.weights[-1].shape[1], num_kernels, kernel_size, kernel_size))

        self.weights.append([np.random.rand(kernel_size, kernel_size)-0.5 for x in range(num_kernels)])
        self.biases.append(np.random.rand(num_kernels)-0.5)
        self.activation_functions.append(activation_function)
        self.num_conv_layers += 1

        input_vector_size = self.neurons_structure[-1][0]
        feature_map_size = ((input_vector_size + 2 * padding_size - kernel_size) // stride) + 1
        self.neurons_structure.append((feature_map_size, feature_map_size))

    def add_dense_layer(
            self,
            num_neurons: int,
            prev_layer_num_neurons: int,
            activation_function: callable,
    ):
        num_prev_layer_neurons = len(np.flatten())
        self.weights.append(np.random.rand(num_neurons, prev_layer_num_neurons)-0.5)
        self.biases.append(np.random.rand(num_neurons)-0.5)
        self.activation_functions.append(activation_function)
        self.num_dense_layers += 1

    def flatten_array(self, array):
        array.flatten()

    def forward_prop(self, input_image: np.array, num_classes: int):
        padding_size = 0
        input_channel_initial_width = input_image.shape[0]
        stride = 1

        for layer_index in range(self.num_conv_layers):
            feature_maps = []  # output channels (z values, then activation values)
            kernel_shape = self.weights[layer_index][0].shape # (weights)
            kernel_size = kernel_shape[1]
            kernel_depth = None
            if len(kernel_shape) > 2:
                kernel_depth = kernel_shape[0]
            feature_map_size = ((input_channel_initial_width + 2*padding_size - kernel_size) // stride) + 1
            num_output_channels = len(self.weights[layer_index])
            for output_channel_index in range(num_output_channels):
                feature_map = np.zeros((feature_map_size, feature_map_size))
                for feature_map_x in range(feature_map_size):
                    for feature_map_y in range(feature_map_size):
                        feature_map_z_value = 0
                        for kernel_x in range(kernel_size):
                            for kernel_y in range(kernel_size):
                                if kernel_depth:
                                    for kernel_z in range(kernel_depth):
                                        # use feature maps, and make 3D. dot product with same weight.
                                feature_map_z_value += (
                                        input_image[feature_map_x + kernel_x][feature_map_y + kernel_y] *
                                        self.weights[layer_index][output_channel_index][kernel_y][kernel_x]
                                )
                        feature_map_z_value += self.biases[layer_index][output_channel_index]
                        feature_map_activation_value = self.activation_functions[layer_index](feature_map_z_value)
                        feature_map[feature_map_x][feature_map_y] = feature_map_activation_value
                feature_maps.append(feature_map)
            feature_maps = np.array(feature_maps)
            self.activations.append(feature_maps)

        self.activations.append(self.activations[-1].flatten())
        num_flattened_neurons = len(self.activations[-1].flatten())

        for layer_index in range(self.num_conv_layers-1, self.num_conv_layers-1 + self.num_dense_layers):
            print(layer_index)
        for class_neuron_index in range(num_classes):
            for flattened_array_neuron_index in range(num_flattened_neurons):
                input_activation_value = self.activations[-1][flattened_array_neuron_index]


        # for flattened_neuron_index, initial_activation_value in enumerate(self.activations[-1].flatten()):
        #     for class_neuron_index in range(num_classes):
        #         weight = self.weights[-1][class_neuron_index][flattened_neuron_index]
        #         bias = self.biases
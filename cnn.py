import numpy as np

from activation_functions import ReLU, sigmoid


class CNN:

    def __init__(self, input_image_shape: tuple):
        self.input_image_shape = input_image_shape

        self.weights = []
        self.biases = []

        self.z_values = []
        self.activation_functions = []
        self.activation_values = []

        self.num_conv_layers = 0
        self.num_flatten_layers = 0
        self.num_dense_layers = 0

        self.layer_names = []

        self.weight_gradients = []
        self.bias_gradients = []

    def add_conv_layer(
            self,
            activation_function: callable,
            num_kernels: int,
            kernel_width: int,
            kernel_height: int,
            stride: int = 1,
            padding: int = 0,
    ):
        if self.weights == []:
            # Number of channels in input image if this is the first layer of the network.
            num_input_channels, input_feature_map_height, input_feature_map_width = self.input_image_shape
        else:
            # Number of output channels in previous layer.
            num_input_channels, input_feature_map_height, input_feature_map_width = self.activation_values[-1].shape

        output_feature_map_height = input_feature_map_height - kernel_height + 1
        output_feature_map_width = input_feature_map_width - kernel_width + 1

        # Add random initialised weights to the network for the new layer.
        self.weights.append(np.random.rand(num_kernels, num_input_channels, kernel_height, kernel_width) - 0.5)

        # Add random initialised biases for the new layer.
        self.biases.append(np.random.rand(num_kernels) - 0.5)

        # Add 0 initialised z_values for the new layer.
        self.z_values.append(np.zeros((num_kernels, output_feature_map_height, output_feature_map_width)))

        # Add activation function for the new layer.
        self.activation_functions.append(activation_function)

        # Add 0 initialised activation_values for the new layer.
        self.activation_values.append(np.zeros((num_kernels, output_feature_map_height, output_feature_map_width)))

        self.num_conv_layers += 1
        self.layer_names.append("conv")

        # Add 0 initialised weight gradients to the network for the new layer.
        self.weight_gradients.append(np.random.rand(num_kernels, num_input_channels, kernel_height, kernel_width) - 0.5)

        # Add 0 initialised bias gradients to the network for the new layer.
        self.bias_gradients.append(np.random.rand(num_kernels) - 0.5)

    def add_flatten_layer(self):
        # add for layer_index
        self.weights.append(None)
        self.biases.append(None)
        self.z_values.append(None)
        self.activation_functions.append(None)
        self.weight_gradients.append(None)
        self.bias_gradients.append(None)

        previous_layer_activation_values = self.activation_values[-1]
        flattened_activation_values = previous_layer_activation_values.flatten()
        self.activation_values.append(flattened_activation_values)

        self.num_flatten_layers += 1
        self.layer_names.append("flatten")

    def add_dense_layer(self, num_neurons, activation_function):
        num_input_neurons = len(self.activation_values[-1])
        self.weights.append(np.random.rand(num_neurons, num_input_neurons)-0.5)
        self.biases.append(np.random.rand(num_neurons)-0.5)

        self.z_values.append(np.zeros(num_neurons))
        self.activation_functions.append(activation_function)
        self.activation_values.append(np.zeros(num_neurons))

        self.num_dense_layers += 1
        self.layer_names.append("dense")

        self.weight_gradients.append(np.random.rand(num_neurons, num_input_neurons)-0.5)
        self.bias_gradients.append(np.random.rand(num_neurons)-0.5)

    # def forward_prop(self, input_images: np.array, class_values: np.array, num_classes: int = 10, print_costs: bool = True):
    def forward_prop(self, input_image: np.array):
        # for image_index, image in enumerate(input_images):
        if len(input_image.shape) == 2:
            image = input_image.reshape(1, 28, 28)
        for layer_index, layer_name in enumerate(self.layer_names):

            if layer_name == "conv":

                if layer_index == 0:
                    # if len(image.shape) == 2:
                    #     input_feature_map_height, input_feature_map_width = image.shape
                    # if len(image.shape) == 3:
                    _, input_feature_map_height, input_feature_map_width = image.shape
                else:
                    _, input_feature_map_height, input_feature_map_width = self.activation_values[layer_index - 1].shape

                num_kernels, num_input_channels, kernel_height, kernel_width = self.weights[layer_index].shape

                output_feature_map_height = input_feature_map_height - kernel_height + 1
                output_feature_map_width = input_feature_map_width - kernel_width + 1

                for output_feature_map_height_index in range(output_feature_map_height):
                    for output_feature_map_width_index in range(output_feature_map_width):

                        for kernel_index in range(num_kernels):

                            feature_map_value = 0
                            for kernel_height_index in range(kernel_height):
                                for kernel_width_index in range(kernel_width):

                                    feature_map_sub_value = 0
                                    for input_channel_index in range(num_input_channels):
                                        if layer_index == 0:
                                            previous_activation_value = image[input_channel_index][output_feature_map_height_index + kernel_height_index][output_feature_map_width_index + kernel_width_index]
                                        else:
                                            previous_activation_value = self.activation_values[layer_index - 1][input_channel_index][output_feature_map_height_index + kernel_height_index][output_feature_map_width_index + kernel_width_index]
                                        weight = self.weights[layer_index][kernel_index][input_channel_index][kernel_height_index][kernel_width_index]

                                        feature_map_sub_value += previous_activation_value * weight

                                    feature_map_value += feature_map_sub_value

                            z_value = feature_map_value

                            activation_function = self.activation_functions[layer_index]

                            activation_value = activation_function(z_value)

                            self.z_values[layer_index][kernel_index][output_feature_map_height_index][output_feature_map_width_index] = z_value
                            self.activation_values[layer_index][kernel_index][output_feature_map_height_index][output_feature_map_width_index] = activation_value

            if layer_name == "flatten":
                self.activation_values[layer_index] = self.activation_values[-1].flatten()

            if layer_name == "dense":
                num_neurons = len(self.activation_values[layer_index])
                previous_layer_neuron_index = len(self.activation_values[layer_index-1])
                for neuron_index in range(num_neurons):
                    for previous_layer_neuron_index in range(previous_layer_neuron_index):

                        previous_activation_value = self.activation_values[layer_index-1][previous_layer_neuron_index]
                        weight = self.weights[layer_index][neuron_index][previous_layer_neuron_index]
                        bias = self.biases[layer_index][neuron_index]

                        z_value = previous_activation_value * weight + bias

                        activation_function = self.activation_functions[layer_index]

                        activation_value = activation_function(z_value)

                        self.z_values[layer_index][neuron_index] = z_value
                        self.activation_values[layer_index][neuron_index] = activation_value

    def cost(self, class_value, num_classes):
        one_hot_class_value_array = np.zeros(num_classes)
        one_hot_class_value_array[class_value] = 1

        class_probabilities = self.activation_values[-1]
        cost = sum((one_hot_class_value_array - class_probabilities) ** 2)

        return cost

        # if print_costs:
        #
        #     class_value = class_values[image_index]
        #     one_hot_class_value_array = np.zeros(num_classes)
        #     one_hot_class_value_array[class_value] = 1
        #
        #     class_probabilities = self.activation_values[-1]
        #     cost = sum((one_hot_class_value_array - class_probabilities) ** 2)
        #     print(f"Cost: {cost}")

    def back_prop(self, class_value, num_classes):
        one_hot_class_value_array = np.zeros(num_classes)
        one_hot_class_value_array[class_value] = 1

    def forward_and_back_prop(
            self,
            input_images: np.array,
            class_values: np.array,
            num_classes: int = 10,
    ):
        for input_image, class_value in zip(input_images, class_values):
            self.forward_prop(input_image)
            cost = self.cost(class_value=class_value, num_classes=num_classes)
            print(f"Cost: {cost}")

        # self.forward_prop(
        #     input_images=input_images,
        #     class_values=class_values,
        #     num_classes=num_classes,
        #     print_costs=print_costs,
        # )
        # self.back_prop()

import numpy as np


class DNN:
    """ This network will train slowly, it purposely
        propagates forward and backward 1  neuron,
        weight, activation and training example at a
        time in an attempt to make visualisation
        on a web-page more strait-forward. """
    def __init__(self, num_input_neurons, learning_rate=0.1):
        self.layer_structure = [num_input_neurons]

        self.weights = []
        self.biases = []
        self.weight_gradients = []
        self.bias_gradients = []
        self.single_sample_weight_gradients = []
        self.single_sample_bias_gradients = []

        self.activation_functions = []
        self.diff_activation_functions = []

        self.z_values = []
        self.activation_values = []

        self.cost = 0

        self.learning_rate = learning_rate

    def add_layer(self, num_neurons, activation_function, diff_activation_function):
        self.layer_structure.append(num_neurons)
        self.activation_functions.append(activation_function)
        self.diff_activation_functions.append(diff_activation_function)

    def initialise_parameter_arrays(self):
        self.weights = [
            np.random.rand(self.layer_structure[x+1], self.layer_structure[x]) - 0.5
            for x in range(len(self.layer_structure) - 1)
        ]

        self.biases = [
            np.random.rand(self.layer_structure[x + 1]) - 0.5
            for x in range(len(self.layer_structure) - 1)
        ]

        self.weight_gradients = [
            np.zeros((self.layer_structure[x+1], self.layer_structure[x]))
            for x in range(len(self.layer_structure) - 1)
        ]

        self.bias_gradients = [
            np.zeros(self.layer_structure[x + 1])
            for x in range(len(self.layer_structure) - 1)
        ]

        self.activation_values = [
            np.zeros(self.layer_structure[x])
            for x in range(len(self.layer_structure))
        ]

        self.z_values = [
            np.zeros(self.layer_structure[x + 1])
            for x in range(len(self.layer_structure) - 1)
        ]

    def forward_prop(self, training_sample_inputs, training_sample_results):
        single_costs = []
        for training_sample_input, training_sample_result in zip(training_sample_inputs, training_sample_results):
            for input_neuron_index, input_neuron_value in enumerate(training_sample_input):
                self.activation_values[0][input_neuron_index] = input_neuron_value

            for layer_index in range(len(self.layer_structure) - 1):
                for neuron_index in range(self.layer_structure[layer_index]):
                    for next_layer_neuron_index in range(self.layer_structure[layer_index + 1]):
                        activation_value = self.activation_values[layer_index][neuron_index]
                        weight = self.weights[layer_index][next_layer_neuron_index][neuron_index]
                        bias = self.biases[layer_index][next_layer_neuron_index]

                        resulting_z_value = activation_value * weight + bias

                        activation_function = self.activation_functions[layer_index]

                        resulting_activation_value = activation_function(resulting_z_value)

                        self.z_values[layer_index][next_layer_neuron_index] = resulting_z_value
                        self.activation_values[layer_index + 1][next_layer_neuron_index] = resulting_activation_value
            single_cost = sum((self.activation_values[-1] - training_sample_result) ** 2) / len(training_sample_result)
            single_costs.append(single_cost)
        total_cost = sum(single_costs) / len(single_costs)
        self.cost = total_cost

    def dCda(self, layer_index, neuron_index, training_sample_index, training_sample_results):
        if layer_index >= len(self.layer_structure) - 1:
            return 2 * (self.activation_values[layer_index][neuron_index] - training_sample_results[training_sample_index][
                neuron_index])

        value = 0
        for next_layer_neuron_index in range(self.layer_structure[layer_index + 1]):
            weight_to_next_layer_neuron = self.weights[layer_index][next_layer_neuron_index][neuron_index]
            diff_activation_function = self.diff_activation_functions[layer_index]
            next_layer_z_value = self.z_values[layer_index][next_layer_neuron_index]
            value += (weight_to_next_layer_neuron
                      * diff_activation_function(next_layer_z_value)
                      * self.dCda(
                        layer_index=layer_index + 1,
                        neuron_index=next_layer_neuron_index,
                        training_sample_index=training_sample_index,
                        training_sample_results=training_sample_results,
                    ))
        return value

    def dCdw(self, layer_index, prev_layer_neuron_index, neuron_index, training_sample_index, training_sample_results):
        neuron_activation = self.activation_values[layer_index - 1][prev_layer_neuron_index]
        diff_activation_function = self.diff_activation_functions[layer_index - 1](self.z_values[layer_index - 1][neuron_index])
        dCda_value = self.dCda(
            layer_index=layer_index,
            neuron_index=neuron_index,
            training_sample_index=training_sample_index,
            training_sample_results=training_sample_results
        )
        return neuron_activation * diff_activation_function * dCda_value

    def dCdb(self, layer_index, neuron_index, training_sample_index, training_sample_results):
        diff_activation_function = self.diff_activation_functions[layer_index - 1](self.z_values[layer_index - 1][neuron_index])
        dCda_value = self.dCda(
            layer_index=layer_index,
            neuron_index=neuron_index,
            training_sample_index=training_sample_index,
            training_sample_results=training_sample_results,
        )
        return diff_activation_function * dCda_value

    def update_weight_and_bias_gradient_values(self, training_sample_inputs, training_sample_results):
        self.single_sample_weight_gradients = [np.array([
            np.zeros((self.layer_structure[x+1], self.layer_structure[x]))
            for x in range(len(self.layer_structure) - 1)
        ]) for t in range(len(training_sample_results))]

        self.single_sample_bias_gradients = [np.array([
            np.zeros(self.layer_structure[x + 1])
            for x in range(len(self.layer_structure) - 1)
        ]) for t in range(len(training_sample_results))]

        for training_sample_index in range(training_sample_inputs.shape[0]):
            for layer_index in range(1, len(self.layer_structure)):
                for neuron_index in range(self.layer_structure[layer_index]):
                    for prev_layer_neuron_index in range(self.layer_structure[layer_index - 1]):
                        self.single_sample_weight_gradients[training_sample_index][layer_index - 1][neuron_index][prev_layer_neuron_index] = self.dCdw(
                            layer_index=layer_index,
                            prev_layer_neuron_index=prev_layer_neuron_index,
                            neuron_index=neuron_index,
                            training_sample_index=training_sample_index,
                            training_sample_results=training_sample_results,
                        )
                    self.single_sample_bias_gradients[training_sample_index][layer_index - 1][neuron_index] = self.dCdb(
                        layer_index=layer_index,
                        neuron_index=neuron_index,
                        training_sample_index=training_sample_index,
                        training_sample_results=training_sample_results,
                    )
        self.weight_gradients = sum(self.single_sample_weight_gradients)
        self.bias_gradients = sum(self.single_sample_bias_gradients)

    def update_weights_and_biases(self):
        for weight_array, weight_gradient_array in zip(self.weights, self.weight_gradients):
            weight_array -= (weight_gradient_array * self.learning_rate)

        for bias_array, bias_gradient_array in zip(self.biases, self.bias_gradients):
            bias_array -= (bias_gradient_array * self.learning_rate)

    def train(self, training_sample_inputs, training_sample_results):
        self.forward_prop(
            training_sample_inputs=training_sample_inputs,
            training_sample_results=training_sample_results,
        )

        self.update_weight_and_bias_gradient_values(
            training_sample_inputs=training_sample_inputs,
            training_sample_results=training_sample_results,
        )

        self.update_weights_and_biases()

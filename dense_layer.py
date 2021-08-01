import numpy as np
import functions.function_utils


class DenseLayer:
    def __init__(self, input_num: int, neuron_num: int, activation):
        self.weights = np.random.randn(input_num, neuron_num) * 0.01
        self.biases = np.zeros((1, neuron_num))
        self.activation = activation()
        self.inputs = None
        self.output = None
        self.weights_der = None
        self.biases_der = None
        self.inputs_der = None

    def forward_pass(self, inputs):
        self.inputs = inputs

        self.activation.forward(np.dot(self.inputs, self.weights) + self.biases)
        self.output = self.activation.output

    def backward_pass(self, prev_layer_grads):
        '''
        Calculates partial derivatives w.r.t weights, biases and inputs
        '''

        self.weights_der = np.dot(self.inputs.T, prev_layer_grads)
        self.biases_der = np.sum(prev_layer_grads, axis=0, keepdims=True)
        self.inputs_der = np.dot(prev_layer_grads, self.weights.T)

    def __str__(self):
        return f"Dense network layer. Neuron number: {self.weights.shape[0]}. Input size: {self.weights.shape[1]}" \
               f" Activation function: {str(self.activation)}"
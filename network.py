from dense_layer import DenseLayer
from functions.function_utils import calc_accuracy


class NeuralNetwork:
    def __init__(self, epochs: int, loss_function, optimizer):
        self.epochs = epochs
        self.loss_function = loss_function()
        self.optimizer = optimizer()
        self.layers = []

    def add(self, layer: DenseLayer):
        self.layers.append(layer)

    def train(self, inputs, target, one_hot=False):
        for epoch_num in range(self.epochs):
            prev_input = inputs

            for ind, layer in enumerate(self.layers):
                layer.forward_pass(prev_input)
                prev_input = layer.output

            loss = self.loss_function.forward(prev_input, target, one_hot)
            accuracy = calc_accuracy(prev_input, target, one_hot=one_hot)

            if epoch_num % 100 == 0:
                print(f"Epoch {epoch_num} Loss: {loss} Accuracy: {accuracy} Learn rate: {self.optimizer.curr_learn_rate}")

            output_layer = self.layers[-1]
            self.loss_function.backward(output_layer.output, target)
            output_layer.backward_pass(self.loss_function.input_der)
            prev_output = output_layer.inputs_der

            for layer in list(reversed(self.layers))[1:]:
                layer.activation.backward(prev_output, target, one_hot)
                layer.backward_pass(layer.activation.input_der)
                prev_output = layer.inputs_der

            for layer in self.layers:
                self.optimizer.update_learn(epoch_num)
                self.optimizer.update(layer)

    def __str__(self):
        display_msg = f"Neural Network. {len(self.layers)} layers. Loss function: {str(self.loss_function)} " \
                      f"Optimizer: {str(self.optimizer)}\n"

        for layer in self.layers:
            display_msg += f"Network layer: {str(layer)} \n"

        return display_msg


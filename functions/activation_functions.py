import numpy as np
from functions.function_utils import SharedSoftmaxCrossEntropy


class Linear:
    def __init__(self):
        self.inputs = None
        self.output = None
        self.input_der = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = self.inputs

    def backward(self, d_gradients):
        self.input_der = d_gradients.copy()

    def __str__(self):
        return "Linear activation function"


class RectifiedLinear:
    def __init__(self):
        self.inputs = None
        self.output = None
        self.input_der = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, self.inputs)

    def backward(self, prev_layer_grads, *args):
        self.input_der = prev_layer_grads.copy()
        self.input_der[self.inputs <= 0] = 0

    def __str__(self):
        return "Rectified linear activation function"


class Softmax(SharedSoftmaxCrossEntropy):
    def __init__(self):
        super().__init__()
        self.inputs = None
        self.output = None
        self.input_der = None

    def forward(self, inputs):
        self.inputs = inputs
        exp = np.exp(inputs - np.max(self.inputs, axis=1, keepdims=True))
        prob = exp / np.sum(exp, axis=1, keepdims=True)
        self.output = prob

    def backward(self, prev_layer_grads, y_true, one_hot=False):
        return super().backward(prev_layer_grads, y_true, one_hot)

    def __str__(self):
        return "Softmax activation function"

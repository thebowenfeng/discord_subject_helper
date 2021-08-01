import numpy as np


def calc_accuracy(pred, targets, one_hot=False):
    pred = np.argmax(pred, axis=1)

    if one_hot:
        targets = np.argmax(targets, axis=1)

    return np.mean(pred == targets)


class SharedSoftmaxCrossEntropy:
    '''
    Optimized class for calculating derivatives
    '''

    def __init__(self):
        self.output = None
        self.input_der = None

    def backward(self, prev_layer_grads, y_true, one_hot=False):
        samples = len(prev_layer_grads)

        if one_hot:
            y_true = np.argmax(y_true, axis=1)

        self.input_der = prev_layer_grads.copy()
        self.input_der[range(samples), y_true] -= 1
        self.input_der = self.input_der / samples

    def __str__(self):
        return "Shared class between Softmax activation and Categorical Cross-Entropy loss"



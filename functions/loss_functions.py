import numpy as np
from functions.function_utils import SharedSoftmaxCrossEntropy


class CategoricalCrossEntropy(SharedSoftmaxCrossEntropy):
    def __init__(self):
        super().__init__()
        self.input_der = None

    def forward(self, pred, target, one_hot=False):
        clipped = np.clip(pred, 1e-7, 1 - 1e-7)

        if one_hot:
            confidences = np.sum(clipped * target, axis=1)
        else:
            confidences = clipped[range(len(pred)), target]

        losses = -np.log(confidences)
        self.output = np.mean(losses)
        return np.mean(losses)

    def backward(self, prev_layer_grads, y_true, one_hot=False):
        return super().backward(prev_layer_grads, y_true, one_hot)

    def __str__(self):
        return "Categorical Cross-Entropy loss function"

import numpy as np

import dense_layer

class Dumb:
    @staticmethod
    def adjust_params(layers: list) -> list:
        for layer in layers:
            layer.weights += 0.05 * np.random.randn(layer.weights.shape[0], layer.weights.shape[1])
            layer.biases += 0.05 * np.random.randn(layer.biases.shape[0], layer.biases.shape[1])

        return layers

    def __call__(self, lowest_loss: int, loss: int, layers: list, best_weights: list):
        if loss < lowest_loss:
            for index, layer in enumerate(layers):
                best_weights[index] = (layer.weights.copy(), layer.biases.copy())
            lowest_loss = loss
        else:
            for index, layer in enumerate(layers):
                layer.weights = best_weights[index][0].copy()
                layer.biases = best_weights[index][1].copy()

        return layers, best_weights, lowest_loss

    def __str__(self):
        return "Dumb optimizer"


class StochasticGradientDescent:
    def __init__(self, learn_rate: float = 1, decay: float = 1e-4, momentum: float = 0.1):
        self.learn_rate = learn_rate
        self.curr_learn_rate = learn_rate
        self.decay_rate = decay
        self.momentum = momentum

    def update(self, layer: dense_layer.DenseLayer):
        if not hasattr(layer, 'weight_momentum'):
            layer.weight_momentum = np.zeros_like(layer.weights)
            layer.bias_momentum = np.zeros_like(layer.biases)

        new_weights = self.momentum * layer.weight_momentum - self.curr_learn_rate * layer.weights_der
        layer.weight_momentum = new_weights
        new_biases = self.momentum * layer.bias_momentum - self.curr_learn_rate * layer.biases_der
        layer.bias_momentum = new_biases

        layer.weights += new_weights
        layer.biases += new_biases

    def update_learn(self, epoch: int):
        self.curr_learn_rate = self.learn_rate * (1 / (1 + self.decay_rate * epoch))

    def __str__(self):
        return "Stochastic gradient descent optimizer"


class AdaptiveGradient:
    def __init__(self, learn_rate: float = 1, decay: float = 1e-4, epsilon: float = 1e-7):
        self.learn_rate = learn_rate
        self.curr_learn_rate = learn_rate
        self.decay_rate = decay
        self.epsilon = epsilon

    def update(self, layer: dense_layer.DenseLayer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.weights_der ** 2
        layer.bias_cache += layer.biases_der ** 2

        layer.weights -= self.curr_learn_rate * layer.weights_der / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases -= self.curr_learn_rate * layer.biases_der / (np.sqrt(layer.bias_cache) + self.epsilon)

    def update_learn(self, epoch: int):
        self.curr_learn_rate = self.learn_rate * (1 / (1 + self.decay_rate * epoch))

    def __str__(self):
        return "Adaptive gradient optimizer"


class RootMeanSquareProp:
    def __init__(self, learn_rate: float = 0.001, decay: float = 0, epsilon: float = 1e-7, rho: float = 0.9):
        self.learn_rate = learn_rate
        self.curr_learn_rate = learn_rate
        self.decay_rate = decay
        self.epsilon = epsilon
        self.rho = rho

    def update(self, layer: dense_layer.DenseLayer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.weights_der ** 2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.biases_der ** 2

        layer.weights -= self.curr_learn_rate * layer.weights_der / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases -= self.curr_learn_rate * layer.biases_der / (np.sqrt(layer.bias_cache) + self.epsilon)

    def update_learn(self, epoch: int):
        self.curr_learn_rate = self.learn_rate * (1 / (1 + self.decay_rate * epoch))

    def __str__(self):
        return "Root mean squared propagation optimizer"


class AdaptiveMomentum:
    def __init__(self, learn_rate: float = 0.05, decay: float = 5e-7, epsilon: float = 1e-7, beta1: float = 0.9, beta2: float = 0.999):
        self.learn_rate = learn_rate
        self.curr_learn_rate = learn_rate
        self.decay_rate = decay
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.curr_epoch = 0

    def update(self, layer: dense_layer.DenseLayer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.weight_momentum = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            layer.bias_momentum = np.zeros_like(layer.biases)

        layer.weight_momentum = self.beta1 * layer.weight_momentum + (1 - self.beta1) * layer.weights_der
        layer.bias_momentum = self.beta1 * layer.bias_momentum + (1 - self.beta1) * layer.biases_der
        w_momentum_corr = layer.weight_momentum / (1 - self.beta1 ** (self.curr_epoch + 1))
        b_momentum_corr = layer.bias_momentum / (1 - self.beta1 ** (self.curr_epoch + 1))

        layer.weight_cache = self.beta2 * layer.weight_cache + (1 - self.beta2) * layer.weights_der ** 2
        layer.bias_cache = self.beta2 * layer.bias_cache + (1 - self.beta2) * layer.biases_der ** 2
        w_cache_corr = layer.weight_cache / (1 - self.beta2 ** (1 + self.curr_epoch))
        b_cache_corr = layer.bias_cache / (1 - self.beta2 ** (1 + self.curr_epoch))

        layer.weights -= self.curr_learn_rate * w_momentum_corr / (np.sqrt(w_cache_corr) + self.epsilon)
        layer.biases -= self.curr_learn_rate * b_momentum_corr / (np.sqrt(b_cache_corr) + self.epsilon)

    def update_learn(self, epoch: int):
        self.curr_learn_rate = self.learn_rate * (1 / (1 + self.decay_rate * epoch))
        self.curr_epoch = epoch

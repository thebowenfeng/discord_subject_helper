from network import NeuralNetwork
from dense_layer import DenseLayer
from functions.activation_functions import RectifiedLinear, Softmax
from functions.loss_functions import CategoricalCrossEntropy
from functions.optimizer_functions import AdaptiveMomentum

#data, classification = spiral_data(samples=100, classes=3)

nn = NeuralNetwork(
    epochs=100001,
    loss_function=CategoricalCrossEntropy,
    optimizer=AdaptiveMomentum
)

nn.add(DenseLayer(
    input_num=2,
    neuron_num=64,
    activation=RectifiedLinear
))

nn.add(DenseLayer(
    input_num=64,
    neuron_num=3,
    activation=Softmax
))

#nn.train(data, classification, one_hot=False)

# pyNeuralNet

A dense neural network built from scratch, using Numpy and Python. Supports a variety of popular optimizers.

### Usage

A demo can be found in `main.py`. 

To use, simply import `NeuralNetwork` from `network.py`, and `DenseLayer` from `dense_layer.py`

Utilize the in-built activation, loss and optimizer functions by checking out all the available functions within the function
folder, and import each as needed. A list of all available function and where to import them can be found below.

Initialize the network by creating an object of NeuralNetwork. Add a layer using `your_neural_network.add()` and pass in a DenseLayer object. Please make sure to set appropriate
attributes when initializing the objects such as activation functions or loss functions. Again, see `main.py` for a fully functioning sample neural network.

### List of all available functions

#### Activation functions (functions/activation_functions.py):

- Linear
- ReLU (Rectified Linear)
- Softmax

#### Loss functions (functions/loss_functions.py)

- CategoricalCrossEntropy

#### Optimizers (functions/optimizer_functions.py)

- *Dumb (A rudimentary, poor-performing custom optimizer used as a PoC)
- StochasticGradientDescent (with momentum)
- Adaptive Gradient
- Root Mean Squared Propagation
- Adaptive Momentum

### Future

Plans to add validation data testing functionality, and regression models.

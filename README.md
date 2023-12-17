# MNIST Neural Network

A classifer for hand-drawn digit images from the MNIST dataset implemented in Python. The neural network uses a 28 * 28 = 784 neuron input layer representing the normalized color for each pixel in a 28x28 image, a single ReLU activated hidden layer, and a 10 neuron sigmoid activated output layer representing the probability of each digit being present in the image.

To compare the performance of stochastic and batched gradient descent, `mnist_stochastic` uses a neural network without batched updates, and `mnist_batch` uses a neural network with minibatches.
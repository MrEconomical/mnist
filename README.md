# MNIST Neural Network

A classifer for hand-drawn digit images from the MNIST dataset implemented in Python. The neural network uses a 28 * 28 = 784 neuron input layer representing the normalized color for each pixel in a 28x28 image, a single ReLU activated hidden layer, and a 10 neuron softmax activated output layer representing the probability of each digit being present in the image.

To compare the performance of stochastic and batched gradient descent, `mnist_stochastic` uses a neural network without batched updates, and `mnist_batch` uses a neural network with minibatches.

## Results

Both the stochastic and batch models already achieve a very high average accuracy with a small hidden layer of 40 neurons. After training for 10 epochs on the 60,000 image training data set, the results evaluated using cross-entropy error on the 10,000 image test data set are:
- Stochastic model (batch size 1): 0.0110
- Batch model (batch size 32): 0.0106
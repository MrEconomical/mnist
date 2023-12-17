from load_mnist import train_data, test_data
from mse_batch_model import Model
import numpy as np
import random

# train model over epochs

input_size = 28 * 28
hidden_size = 40
output_size = 10
learning_rate = 0.0015
model = Model.with_random_weights(input_size, hidden_size, output_size, learning_rate)

epochs = 10
batch_size = 32
log_interval = 20000

for e in range(epochs):
    print("running epoch", e + 1)
    random.shuffle(train_data)
    hidden_batch = np.zeros((model.hidden_size, model.input_size + 1))
    output_batch = np.zeros((model.output_size, model.hidden_size + 1))
    total_error = 0

    for i in range(len(train_data)):
        image = train_data[i]
        hidden_output, output = model.forward(image[0])
        hidden_grad, output_grad, output_error = model.back_prop(image[0], hidden_output, output, image[1])

        hidden_batch += hidden_grad
        output_batch += output_grad
        total_error += output_error

        if (i + 1) % batch_size == 0:
            model.apply_gradients(hidden_batch, output_batch)
            hidden_batch.fill(0)
            output_batch.fill(0)

        if (i + 1) % log_interval == 0:
            print("[" + str(i + 1) + "] error:", total_error / log_interval)
            total_error = 0

# test model

print()
print("evaluating model:")
total_error = 0
for image in test_data:
    hidden_output, output = model.forward(image[0])
    difference = image[1] - output
    total_error += difference.dot(difference) / len(difference)
print("error:", total_error / len(test_data))
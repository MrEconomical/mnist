from load_mnist import train_data, test_data
from mse_multi_model import Model
import numpy as np

# train model over epochs

input_size = 28 * 28
hidden_size = 30
output_size = 10
learning_rate = 0.001
model = Model.with_random_weights(input_size, hidden_size, output_size, learning_rate)

epochs = 5
log_interval = 10000

for e in range(epochs):
    print("running epoch", e + 1)
    total_error = 0

    for i in range(len(train_data)):
        image = train_data[i]
        hidden_output, output = model.forward(image[0])
        output_error = model.back_prop(image[0], hidden_output, output, image[1])
        total_error += output_error

        if i % log_interval == 0:
            print("[" + str(i) + "] error:", total_error / log_interval)
            total_error = 0

print()
print("evaluating model:")

for image in test_data[0:5]:
    hidden_output, output = model.forward(image[0])
    print("output:", output, "expected:", image[1])
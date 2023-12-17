import csv
import numpy as np

def parse_row(row):
    image = np.array(row[1:], dtype=np.float64) / 255
    result = np.zeros(10, dtype=np.float64)
    result[int(row[0])] = 1
    return (image, result)

train_data = []
train_file = open("data/mnist_train.csv")
for row in csv.reader(train_file):
    train_data.append(parse_row(row))
print("parsed", len(train_data), "training images")

test_data = []
test_file = open("data/mnist_test.csv")
for row in csv.reader(test_file):
    test_data.append(parse_row(row))
print("parsed", len(test_data), "test images")
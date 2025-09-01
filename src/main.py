import mnist_loader
import network

# Load the MNIST dataset
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

training_data = list(training_data)
validation_data = list(validation_data)
test_data = list(test_data)

# Initialize the network with:
# 784 input neurons (28x28 pixels),
# 30 hidden neurons,
# 10 output neurons (digits 0–9)
net = network.Network([784, 30, 10])

# Train with:
# - 30 epochs
# - mini-batch size of 10
# - learning rate = 3.0
# Also evaluate against test data after each epoch
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
